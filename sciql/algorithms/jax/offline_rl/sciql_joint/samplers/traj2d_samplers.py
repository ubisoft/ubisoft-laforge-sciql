import flax
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable
from sciql.core.data import Sampler
from sciql.utils.flax_utils import nonpytree_field
from sciql.algorithms.jax.offline_rl.sciql_joint.samplers.samplers import MINE_Batch, SCIQL_JOINT_Batch, SCIQL_JOINT_Data, SCIQL_JOINT_Data_Config
from sciql.utils.imports import instantiate_class, get_class, get_arguments

def get_normalization(dataset: SCIQL_JOINT_Data) -> float:
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
    data: SCIQL_JOINT_Data
    
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
    labels_numbers: tuple
    p_curlabel: jnp.ndarray      # [2*N]
    p_trajlabel: jnp.ndarray     # [2*N]
    p_randomlabel: jnp.ndarray   # [2*N]
    geom_lambda: jnp.ndarray     # [2*N]
    dist_max: jnp.ndarray        # [2*N]
    geom_sample: jnp.ndarray     # [2*N] boolean flags (2 per labeler)
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
        ep_ids_np = np.repeat(np.arange(len(lengths_np), dtype=np.int32), lengths_np)      # (N,)

        initial_locs = jnp.asarray(starts_np, dtype=jnp.int32)
        final_locs = jnp.asarray(finals_np, dtype=jnp.int32)
        ep_id_of_index = jnp.asarray(ep_ids_np, dtype=jnp.int32)

        # Cache history offsets once (descending)
        desc_offsets = jnp.arange(0, -history, -1, dtype=jnp.int32)

        # Dataset
        dataset = SCIQL_JOINT_Data(
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

        # Sampling params (broadcast scalars to per-labeler arrays)
        config = SCIQL_JOINT_Data_Config(**cfg)

        def as_array_per_label(x, N):
            return jnp.array(x) if isinstance(x, Iterable) else jnp.array([x]*N)

        N = len(labels_numbers)
        value_p_curlabel   = as_array_per_label(config.value_p_curlabel,   N).astype(jnp.float32)
        value_p_trajlabel  = as_array_per_label(config.value_p_trajlabel,  N).astype(jnp.float32)
        value_p_randomlabel= as_array_per_label(config.value_p_randomlabel,N).astype(jnp.float32)
        value_geom_lambda  = as_array_per_label(config.value_geom_lambda,  N).astype(jnp.float32)
        value_dist_max     = as_array_per_label(config.value_dist_max,     N).astype(jnp.int32)
        value_geom_sample  = as_array_per_label(config.value_geom_sample,  N).astype(jnp.bool_)

        actor_p_curlabel   = as_array_per_label(config.actor_p_curlabel,   N).astype(jnp.float32)
        actor_p_trajlabel  = as_array_per_label(config.actor_p_trajlabel,  N).astype(jnp.float32)
        actor_p_randomlabel= as_array_per_label(config.actor_p_randomlabel,N).astype(jnp.float32)
        actor_geom_lambda  = as_array_per_label(config.actor_geom_lambda,  N).astype(jnp.float32)
        actor_dist_max     = as_array_per_label(config.actor_dist_max,     N).astype(jnp.int32)
        actor_geom_sample  = as_array_per_label(config.actor_geom_sample,  N).astype(jnp.bool_)

        p_curlabel = jnp.concatenate([value_p_curlabel, actor_p_curlabel])
        p_trajlabel = jnp.concatenate([value_p_trajlabel, actor_p_trajlabel])
        p_randomlabel = jnp.concatenate([value_p_randomlabel, actor_p_randomlabel])
        geom_lambda = jnp.concatenate([value_geom_lambda, actor_geom_lambda])
        dist_max = jnp.concatenate([value_dist_max, actor_dist_max])
        geom_sample = jnp.concatenate([value_geom_sample, actor_geom_sample])

        assert len(p_curlabel) == 2*N
        assert len(p_trajlabel) == 2*N
        assert len(p_randomlabel) == 2*N
        assert len(geom_lambda) == 2*N
        assert len(dist_max) == 2*N
        assert len(geom_sample) == 2*N

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
            labels_probs=labels_probs,
            p_curlabel=p_curlabel,
            p_trajlabel=p_trajlabel,
            p_randomlabel=p_randomlabel,
            geom_lambda=geom_lambda,
            dist_max=dist_max,
            geom_sample=geom_sample,
        )

    @jax.jit
    def get_goal_idxs(
        self,
        curr_idxs: jnp.ndarray,       # [B]
        final_idxs: jnp.ndarray,      # [B]
        p_curgoal: jnp.ndarray,       # scalar (per labeler via vmap)
        p_trajgoal: jnp.ndarray,      # scalar
        p_randomgoal: jnp.ndarray,    # scalar
        geom_lambda: jnp.ndarray,     # scalar
        geom_sample: jnp.ndarray,     # scalar bool
        dist_max: jnp.ndarray,        # scalar int
        rng: jnp.ndarray,             # PRNGKey
    ) -> jnp.ndarray:                 # [B] indices

        # Get goals sampling rngs
        rng_1, rng_2, rng_3, rng_4 = jax.random.split(rng, 4)
        batch_size = curr_idxs.shape[0]

        # Random goals anywhere in dataset
        random_goal_idxs = jax.random.randint(rng_1, (batch_size,), 0, self.size)

        # Trajectory goals — conditional on geom_sample (JAX-safe)
        def _geom(_):
            offsets = jax.random.geometric(rng_2, p=(1.0 - geom_lambda), shape=(batch_size,))
            offsets = jnp.minimum(offsets.astype(jnp.int32), dist_max)
            return jnp.minimum(curr_idxs + offsets, final_idxs)

        def _nongeom(_):
            distances = jax.random.uniform(rng_2, shape=(batch_size,), dtype=jnp.float32)
            return jnp.round(
                jnp.minimum(curr_idxs + 1, final_idxs) * distances
                + final_idxs * (1.0 - distances)
            ).astype(jnp.int32)

        traj_goal_idxs = jax.lax.cond(geom_sample, _geom, _nongeom, operand=None)

        # Decide between traj vs random when not current
        sum_traj_random = p_trajgoal + p_randomgoal
        prob_traj_given_not_current = jax.lax.cond(
            sum_traj_random > 0.0,
            lambda s: p_trajgoal / s,
            lambda s: jnp.array(1.0, dtype=jnp.float32),
            sum_traj_random,
        )

        choose_traj_over_random = (
            jax.random.uniform(rng_3, (batch_size,), dtype=jnp.float32)
            < prob_traj_given_not_current
        )
        other_goal_idxs = jnp.where(choose_traj_over_random, traj_goal_idxs, random_goal_idxs)

        # Maybe keep the current index
        choose_current = jax.random.uniform(rng_4, (batch_size,), dtype=jnp.float32) < p_curgoal
        final_goal_indices = jnp.where(choose_current, curr_idxs, other_goal_idxs)

        return final_goal_indices.astype(jnp.int32)
    
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
        rng_idxs, rng_labels = jax.random.split(rng, 2)
        rngs_labels = jax.random.split(rng_labels, 2*len(self.labels_numbers))   # [N, 2]

        # Sample starting indices
        curr_idxs = jax.random.randint(rng_idxs, (batch_size,), 0, len(self.data.observations))
        ep_ids = self.ep_id_of_index[curr_idxs]
        start_idxs = self.initial_locs[ep_ids]      # (B,)
        final_idxs = self.final_locs[ep_ids]

        # Sample label indices
        vmapped_get_goal_idxs = jax.jit(
            jax.vmap(
                self.get_goal_idxs,
                in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0),
                out_axes=1,  # stack per-labeler outputs as [B, N]
            )
        )
        all_label_idxs = vmapped_get_goal_idxs(
            curr_idxs,
            final_idxs,
            self.p_curlabel,
            self.p_trajlabel,
            self.p_randomlabel,
            self.geom_lambda,
            self.geom_sample,
            self.dist_max,
            rngs_labels,
        )

        # Gather labels: data.labels is [D, N], indices [B, N] -> [B, N]
        value_label_idxs, actor_label_idxs = jnp.split(all_label_idxs, 2, axis=1)
        value_labels = jnp.take_along_axis(self.data.labels, value_label_idxs, axis=0)
        actor_labels = jnp.take_along_axis(self.data.labels, actor_label_idxs, axis=0)

        # Get observations and next_observations
        observations, next_observations = self.get_observations(start_idxs, curr_idxs)

        batch = SCIQL_JOINT_Batch(
            observations=observations,
            actions=self.data.actions[curr_idxs],
            next_observations=next_observations,
            task_rewards=self.data.rewards[curr_idxs],
            task_masks=1-self.data.terminated[curr_idxs],
            curr_labels=self.data.labels[curr_idxs],
            value_labels=value_labels,
            actor_labels=actor_labels,
            label_masks=1-self.data.terminated[value_labels],
        )
        return batch
    
    def sample_mine_batch(self, batch_size: int, rng:jax.random.PRNGKey = None):

        # Compute rngs
        if rng is None: rng = jax.random.PRNGKey(0)
        rng_idxs, rng_mine = jax.random.split(rng, 2)

        curr_idxs = jax.random.randint(rng_idxs, (batch_size,), 0, len(self.data.observations))
        curr_batch = jax.tree_util.tree_map(lambda x: x[curr_idxs], self.data)

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


if __name__ == '__main__':
    from omegaconf import DictConfig
    from functools import partial
    from tqdm import tqdm

    cfg = DictConfig({
        'episodes_db': {
            'classname': 'sciql.data.episodes_db.traj2d.episodes_db.Traj2D_EpisodesDB',
            'directory': 'traj2d/random_circles-v0'
        },
        'env': {
            'classname': 'sciql.envs.traj2d.envs.Traj2D_Environment'
        },
        'labels': ['position_label', 'turn_direction_label'],
        'seed': 0,
        'history': 4,
        'include_start_state': False,
        'clip_to_eps': False,
        'normalize_state': False,
        'normalize_reward_by_rewards': False,
        'shift_rewards': False,
        'normalize_reward_by_returns': False,
        'terminate_on_end': False
    })

    sampler = Traj2D_Sampler.create(cfg)
    sample_fn = jax.jit(partial(sampler.sample_batch, batch_size=1024))

    # Warmup
    _ = sample_fn()

    # Tight loop (avoid prints in the loop)
    for _ in tqdm(range(1000000)):
        batch = sample_fn()
    print(batch.observations.shape)
