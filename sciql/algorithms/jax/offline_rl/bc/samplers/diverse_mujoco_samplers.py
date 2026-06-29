import os
import flax
import jax
import jax.numpy as jnp
from tqdm import tqdm
from sciql.core.data import Sampler
from sciql.utils.flax_utils import nonpytree_field
from sciql.algorithms.jax.offline_rl.bc.samplers.samplers import BC_Data, BC_Batch
from sciql.utils.imports import instantiate_class

@flax.struct.dataclass
class DiverseMujoco_Sampler(Sampler):

    # Data
    data: BC_Data
    
    # Episode metadata
    obs_mean: float = nonpytree_field()
    obs_std: float = nonpytree_field()

    @classmethod
    def create(cls, cfg):
        
        # Get EpisodesDB and Env
        episodes_db = instantiate_class(cfg.episodes_db)

        # Prepare data
        observations_list = []
        actions_list = []

        # Compute data
        for episode in tqdm(episodes_db, desc='Loading episodes'):
            episode_dict = episode.to_dict()
            observations_list.append(episode_dict['observation/full'])
            actions_list.append(episode_dict['action/full'])

        dataset = BC_Data(
            observations=jnp.concatenate(observations_list, axis=0).astype(jnp.float32),
            actions=jnp.concatenate(actions_list, axis=0).astype(jnp.float32)
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
            obs_mean=float(obs_mean), 
            obs_std=float(obs_std)
        )
    
    def sample_batch(self, batch_size: int, rng:jax.random.PRNGKey = None) -> BC_Batch:
        if rng is None: rng = jax.random.PRNGKey(0)
        batch_indices = jax.random.randint(rng, (batch_size,), 0, len(self.data.observations))
        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], self.data)
        return batch

    def __len__(self):
        return len(self.data.observations)
        
        
