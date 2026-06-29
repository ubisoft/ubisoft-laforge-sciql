import flax
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from sciql.core.data import Sampler
from sciql.utils.flax_utils import nonpytree_field
from sciql.algorithms.jax.offline_rl.cbc.samplers.samplers import CBC_Data, CBC_Batch
from sciql.utils.imports import instantiate_class

@flax.struct.dataclass
class DiverseMujoco_Sampler(Sampler):

    # Data
    data: CBC_Data

    # Episode metadata (for boundary-safe windowing)
    obs_mean: float = nonpytree_field()
    obs_std: float = nonpytree_field()

    # Stats
    labels_numbers: tuple

    @classmethod
    def create(cls, cfg):
        
        # Get EpisodesDB and Env
        episodes_db = instantiate_class(cfg.episodes_db)
        env = instantiate_class(cfg.env)

        # Prepare data
        observations_list = []
        actions_list = []
        labels_list = []

        # Get labels and labels info
        labels_names = [label_name for label_name in cfg.labels]
        labels_numbers = [env.labels[name].num_labels for name in labels_names]

        # Compute data
        for episode in tqdm(episodes_db, desc='Loading episodes'):
            episode_dict = episode.to_dict()
            observations_list.append(episode_dict['observation/full'])
            actions_list.append(episode_dict['action/full'])
            labels_list.append(np.stack([env.labels[name](episode)[name] for name in labels_names], axis=-1))

        dataset = CBC_Data(
            observations=jnp.concatenate(observations_list, axis=0).astype(jnp.float32),
            actions=jnp.concatenate(actions_list, axis=0).astype(jnp.float32),
            labels=jnp.concatenate(labels_list, axis=0).astype(jnp.int32)
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
            obs_std=float(obs_std), 
            labels_numbers=tuple(labels_numbers)
        )
    
    def sample_batch(self, batch_size: int, rng:jax.random.PRNGKey = None) -> CBC_Batch:
        if rng is None: rng = jax.random.PRNGKey(0)
        batch_indices = jax.random.randint(rng, (batch_size,), 0, len(self.data.observations))
        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], self.data)
        return batch

    def __len__(self):
        return len(self.data.observations)
        
        
