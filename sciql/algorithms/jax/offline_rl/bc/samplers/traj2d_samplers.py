import flax
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from sciql.core.data import Sampler
from sciql.utils.flax_utils import nonpytree_field
from sciql.algorithms.jax.offline_rl.bc.samplers.samplers import BC_Data, BC_Batch
from sciql.utils.imports import instantiate_class

@flax.struct.dataclass
class Traj2D_Sampler(Sampler):

    # Data
    data: BC_Data

    # Episode metadata (for boundary-safe windowing)
    size: int = nonpytree_field()
    initial_locs: jnp.ndarray          # (E,) start index of each episode
    final_locs: jnp.ndarray            # (E,) last index (inclusive) of each episode
    ep_id_of_index: jnp.ndarray        # (N,) episode id for each global index
    obs_mean: float
    obs_std: float

    # History metadata
    desc_offsets: jnp.ndarray          # (H,) cached offsets for history order [t, t-1, ...]
    history: int = nonpytree_field()
    include_start_state: bool = nonpytree_field()

    @classmethod
    def create(cls, cfg):
        
        # History config
        history: int = int(cfg.history)
        include_start_state: bool = bool(cfg.include_start_state)
        assert history >= 1

        # Get EpisodesDB and Env
        episodes_db = instantiate_class(cfg.episodes_db)

        # Prepare data
        observations_list = []
        actions_list = []
        lengths_np = []

        # Compute data
        for episode in tqdm(episodes_db, desc='Loading episodes'):
            episode_dict = episode.to_dict()
            observations_list.append(episode_dict['observation/full'])
            actions_list.append(episode_dict['action/full'])
            lengths_np.append(len(episode_dict['observation/full']))
        
        # Episode indices from lengths (avoids per-batch search)
        lengths_np = np.asarray(lengths_np, dtype=np.int32)
        starts_np = np.concatenate([[0], np.cumsum(lengths_np[:-1])]).astype(np.int32)     # (E,)
        finals_np = (starts_np + lengths_np - 1).astype(np.int32)                          # (E,) inclusive
        ep_ids_np = np.repeat(np.arange(len(lengths_np), dtype=np.int32), lengths_np)      # (N,)

        initial_locs = jnp.asarray(starts_np, dtype=jnp.int32)
        final_locs = jnp.asarray(finals_np, dtype=jnp.int32)
        ep_id_of_index = jnp.asarray(ep_ids_np, dtype=jnp.int32)

        # Cache history offsets once (descending)
        desc_offsets = jnp.arange(0, -history, -1, dtype=jnp.int32)

        # Dataset
        dataset = BC_Data(
            observations=jnp.concatenate(observations_list, axis=0).astype(jnp.float32),
            actions=jnp.concatenate(actions_list, axis=0).astype(jnp.float32),
        )

        # Clip actions
        if cfg.clip_to_eps:
            lim = 1 - cfg.eps
            dataset = dataset._replace(
                actions = jnp.clip(dataset.actions, -lim, lim)
            )
            
        # Normalize states
        obs_mean, obs_std = 0, 1
        if cfg.normalize_state:
            obs_mean = dataset.observations.mean(0)
            obs_std = dataset.observations.std(0)
            dataset = dataset._replace(
                observations=(dataset.observations - obs_mean) / (obs_std + 1e-5),
                next_observations=(dataset.next_observations - obs_mean) / (obs_std + 1e-5),
            )

        return cls(
            data=dataset,
            size=dataset.observations.shape[0],
            initial_locs=initial_locs,
            final_locs=final_locs,
            ep_id_of_index=ep_id_of_index,
            obs_mean=float(obs_mean),
            obs_std=float(obs_std),
            desc_offsets=desc_offsets,
            history=history,
            include_start_state=include_start_state
        )
    
    def get_observations(self, start_idxs, curr_idxs):

        # Observation history indices: [t, t-1, ..., t-H+1], clamped to s_0
        raw_hist = curr_idxs[:, None] + self.desc_offsets[None, :]    # (B, H)
        hist_idx = jnp.maximum(raw_hist, start_idxs[:, None])    # (B, H)
        obs_hist = self.data.observations[hist_idx]             # (B, H, D)
        B_, H_, D = obs_hist.shape
        obs_flat = obs_hist.reshape(B_, H_ * D)

        if self.include_start_state:
            s0 = self.data.observations[start_idxs]                # (B, D)
            obs_feat  = jnp.concatenate([obs_flat,  s0], axis=-1)
        else:
            obs_feat = obs_flat
        
        return obs_feat

    def sample_batch(self, batch_size: int, rng:jax.random.PRNGKey = None) -> BC_Batch:
        
        # RNGs
        if rng is None: rng = jax.random.PRNGKey(0)

        # Sample indices
        curr_idxs = jax.random.randint(rng, (batch_size,), 0, len(self.data.observations))
        ep_ids = self.ep_id_of_index[curr_idxs]
        start_idxs = self.initial_locs[ep_ids]
        
        # Get batch
        batch = jax.tree_util.tree_map(lambda x: x[curr_idxs], self.data)

        # Get observations and next_observations
        observations = self.get_observations(start_idxs, curr_idxs)

        batch = BC_Batch(
            observations=observations,
            actions=batch.actions,
        )
        return batch

    def __len__(self):
        return len(self.data.observations)