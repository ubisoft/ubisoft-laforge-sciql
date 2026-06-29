import flax
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from sciql.core.data import Sampler
from sciql.utils.flax_utils import nonpytree_field
from sciql.algorithms.jax.offline_rl.sorl.samplers.samplers import MINE_Batch, SORL_Batch, SORL_Data
from sciql.utils.imports import instantiate_class

def get_normalization(dataset: SORL_Data) -> float:
    dataset = jax.tree_util.tree_map(lambda x: np.array(x), dataset)
    returns = []
    ret = 0
    dones = np.logical_or(dataset.terminated, dataset.truncated)
    for r, term in zip(dataset.rewards, dones):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000

@flax.struct.dataclass
class Traj2D_Sampler(Sampler):

    # Data
    data: SORL_Data
    
    # Episode metadata (for boundary-safe windowing)
    size: int = nonpytree_field()
    initial_locs: jnp.ndarray          # (E,) start index of each episode
    final_locs: jnp.ndarray            # (E,) last index (inclusive) of each episode
    ep_id_of_index: jnp.ndarray        # (N,) episode id for each global index
    obs_mean: float = nonpytree_field()
    obs_std: float = nonpytree_field()

    # History metadata
    desc_offsets: jnp.ndarray          # (H,) cached offsets for history order [t, t-1, ...]
    history: int = nonpytree_field()
    include_start_state: bool = nonpytree_field()

    # Labeling metadata
    labels_numbers: tuple = nonpytree_field()
    labels_probs: jnp.ndarray = nonpytree_field()

    @classmethod
    def create(cls, cfg):

        # History config
        history: int = int(cfg.history)
        include_start_state: bool = bool(cfg.include_start_state)
        assert history >= 1

        # Episodes DB
        episodes_db = instantiate_class(cfg.episodes_db)
        env = instantiate_class(cfg.env)

        # Load data
        observations_list = []
        actions_list = []
        rewards_list = []
        next_observations_list = []
        terminated_list = []
        truncated_list = []
        labels_list = []
        lengths_np = []

        # Get labels and labels info
        labels_names = [label_name for label_name in cfg.labels]
        labels_numbers = [env.labels[name].num_labels for name in labels_names]

        # Compute data
        for episode in tqdm(episodes_db, desc='Loading episodes'):
            episode_dict = episode.to_dict()
            observations_list.append(episode_dict['observation/full'])
            actions_list.append(episode_dict['action/full'])
            rewards_list.append(episode_dict['reward'])
            next_observations_list.append(episode_dict['next_observation/full'])
            terminated_list.append(episode_dict['terminated'])
            truncated_list.append(episode_dict['truncated'])
            labels_list.append(np.stack([env.labels[name](episode)[name] for name in labels_names], axis=-1))
            lengths_np.append(len(episode_dict['observation/full']))

        # Episode indices from lengths (avoids per-batch search)
        lengths_np = np.asarray(lengths_np, dtype=np.int32)
        starts_np = np.concatenate([[0], np.cumsum(lengths_np[:-1])]).astype(np.int32)     # (E,)
        finals_np = (starts_np + lengths_np - 1).astype(np.int32)                          # (E,) inclusive
        ep_ids_np = np.repeat(np.arange(len(lengths_np), dtype=np.int32), lengths_np)      # (D,)

        initial_locs = jnp.asarray(starts_np, dtype=jnp.int32)
        final_locs = jnp.asarray(finals_np, dtype=jnp.int32)
        ep_id_of_index = jnp.asarray(ep_ids_np, dtype=jnp.int32)

        # Cache history offsets once (descending)
        desc_offsets = jnp.arange(0, -history, -1, dtype=jnp.int32)

        # Dataset
        dataset = SORL_Data(
            observations=jnp.concatenate(observations_list, axis=0).astype(jnp.float32),
            actions=jnp.concatenate(actions_list, axis=0).astype(jnp.float32),
            rewards=jnp.concatenate(rewards_list, axis=0).astype(jnp.float32),
            next_observations=jnp.concatenate(next_observations_list, axis=0).astype(jnp.float32),
            terminated=jnp.concatenate(terminated_list, axis=0).astype(jnp.bool_),
            truncated=jnp.concatenate(truncated_list, axis=0).astype(jnp.bool_),
            labels=jnp.concatenate(labels_list, axis=0).astype(jnp.int32)   # [D, N]
        )

        # Terminate on end
        if cfg.terminate_on_end:
            dones = jnp.logical_or(dataset.terminated, dataset.truncated)
            dataset = dataset._replace(
                terminated = dones
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
            
        # Balance rewards
        if cfg.normalize_reward_by_rewards:
            r_min, r_max = jnp.min(dataset.rewards), jnp.max(dataset.rewards)
            dataset = dataset._replace(rewards = (dataset.rewards - r_min) / (r_max - r_min))

        # Shift rewards
        if cfg.shift_rewards:
            dataset = dataset._replace(rewards=dataset.rewards - 1.0)

        # Scale rewards
        if cfg.normalize_reward_by_returns:    
            normalizing_factor = get_normalization(dataset)
            dataset = dataset._replace(rewards=dataset.rewards / normalizing_factor)

        # Compute the joint probabilities
        D, N = dataset.labels.shape
        C_max = max(labels_numbers)
        # [D, N, C_max], one-hot per label dim (values >= its Ci never occur -> implicit zeros)
        oh = jax.nn.one_hot(dataset.labels, C_max, dtype=jnp.float32)
        # Build joint via broadcasted outer-product over dims, then sum over D
        joint_prod = 1.0
        for n in range(N):
            x = oh[:, n, :]  # [D, C_max]
            x = x.reshape((D,) + (1,) * n + (C_max,) + (1,) * (N - 1 - n))
            joint_prod = joint_prod * x
        counts = joint_prod.sum(axis=0)             # [C_max]*N
        labels_probs  = counts / jnp.maximum(D, 1)         # avoid div-by-zero if D==0

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
            include_start_state=include_start_state,
            labels_numbers=tuple(labels_numbers),
            labels_probs=labels_probs
        )
    
    def get_observations(self, start_idxs, curr_idxs):

        # Observation history indices: [t, t-1, ..., t-H+1], clamped to s_0
        raw_hist = curr_idxs[:, None] + self.desc_offsets[None, :]    # (B, H)
        hist_idx = jnp.maximum(raw_hist, start_idxs[:, None])    # (B, H)
        obs_hist = self.data.observations[hist_idx]             # (B, H, D)
        B_, H_, D = obs_hist.shape
        obs_flat = obs_hist.reshape(B_, H_ * D)

        # Next history: [s_{t+1}] + [s_t, s_{t-1}, ...] (H-1 frames)
        next_first = self.data.next_observations[curr_idxs][:, None, :]     # (B, 1, D)
        next_hist = jnp.concatenate([next_first, obs_hist[:, :-1, :]], axis=1)  # (B, H, D)
        next_flat = next_hist.reshape(B_, H_ * D)

        if self.include_start_state:
            s0 = self.data.observations[start_idxs]                # (B, D)
            obs_feat  = jnp.concatenate([obs_flat,  s0], axis=-1)
            next_feat = jnp.concatenate([next_flat, s0], axis=-1)
        else:
            obs_feat, next_feat = obs_flat, next_flat
        
        return obs_feat, next_feat
        
    def sample_batch(self, batch_size: int, rng: jnp.ndarray = None):

        # RNGs
        if rng is None: rng = jax.random.PRNGKey(0)

        # Sample starting indices
        curr_idxs = jax.random.randint(rng, (batch_size,), 0, len(self.data.observations))
        ep_ids = self.ep_id_of_index[curr_idxs]
        start_idxs = self.initial_locs[ep_ids]

        # Get observations and next_observations
        observations, next_observations = self.get_observations(start_idxs, curr_idxs)

        batch = SORL_Batch(
            observations=observations,
            actions=self.data.actions[curr_idxs],
            next_observations=next_observations,
            task_rewards=self.data.rewards[curr_idxs],
            task_masks=1-self.data.terminated[curr_idxs],
            curr_labels=self.data.labels[curr_idxs],
        )
        return batch
    
    def sample_mine_batch(self, batch_size: int, rng:jax.random.PRNGKey = None):

        # Compute rngs
        if rng is None: rng = jax.random.PRNGKey(0)
        rng_idxs, rng_mine = jax.random.split(rng, 2)

        curr_idxs = jax.random.randint(rng_idxs, (batch_size,), 0, len(self.data.observations))
        curr_batch: SORL_Data = jax.tree_util.tree_map(lambda x: x[curr_idxs], self.data)

        marginal_idxs = jax.random.randint(rng_mine, (batch_size,), 0, len(self.data.observations))
        marginal_labels = self.data.labels[marginal_idxs]
        
        ep_ids = self.ep_id_of_index[curr_idxs]
        start_idxs = self.initial_locs[ep_ids]
        observations, _ = self.get_observations(start_idxs, curr_idxs)

        batch = MINE_Batch(
            observations=observations,
            actions=curr_batch.actions,
            joint_labels=curr_batch.labels,
            marginal_labels=marginal_labels
        )

        return batch

    def __len__(self):
        return int(self.data.observations.shape[0])