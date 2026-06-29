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
class DiverseMujoco_Sampler(Sampler):
    
    # Data
    data: SORL_Data

    # Trajectory metadata
    obs_mean: float = nonpytree_field()
    obs_std: float = nonpytree_field()

    # Labeling stats
    labels_numbers: tuple = nonpytree_field()
    labels_probs: jnp.ndarray = nonpytree_field()
    
    @classmethod
    def create(cls, cfg):

        # Get EpisodesDB and Env
        episodes_db = instantiate_class(cfg.episodes_db)
        env = instantiate_class(cfg.env)

        # Prepare data
        observations_list = []
        actions_list = []
        rewards_list = []
        next_observations_list = []
        terminated_list = []
        truncated_list = []
        labels_list = []

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
            obs_mean=float(obs_mean), 
            obs_std=float(obs_std), 
            labels_numbers=tuple(labels_numbers),
            labels_probs=labels_probs,
        )

    def sample_batch(self, batch_size: int, rng: jnp.ndarray = None):

        # RNGs
        if rng is None: rng = jax.random.PRNGKey(0)

        # Sample batch
        curr_idxs = jax.random.randint(rng, (batch_size,), 0, len(self.data.observations))
        batch = SORL_Batch(
            observations=self.data.observations[curr_idxs],
            actions=self.data.actions[curr_idxs],
            next_observations=self.data.next_observations[curr_idxs],
            task_rewards=self.data.rewards[curr_idxs],
            task_masks=1-self.data.terminated[curr_idxs],
            curr_labels=self.data.labels[curr_idxs]
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

        batch = MINE_Batch(
            observations=curr_batch.observations,
            actions=curr_batch.actions,
            joint_labels=curr_batch.labels,
            marginal_labels=marginal_labels
        )

        return batch

    def __len__(self):
        return len(self.data.observations)
