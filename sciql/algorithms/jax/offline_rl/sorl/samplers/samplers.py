import flax
import jax
import jax.numpy as jnp
from typing import Any, NamedTuple
from sciql.core.data import Sampler
from pydantic import BaseModel

class SORL_Data(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    labels: jnp.ndarray

class SORL_Batch(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    next_observations: jnp.ndarray
    task_rewards: jnp.ndarray
    task_masks: jnp.ndarray
    curr_labels: jnp.ndarray

class MINE_Batch(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    joint_labels: jnp.ndarray
    marginal_labels: jnp.ndarray

@flax.struct.dataclass
class Dummy_Sampler(Sampler):
    """
    A dummy sampler that reconstructs a zero-filled DATA batch from
    shape and dtype metadata, using a simple dictionary comprehension.
    """
    data_structure: Any
    obs_mean: float
    obs_std: float
    labels_numbers: tuple
    labels_probs: list

    def sample_batch(self, batch_size: int, rng: jax.random.PRNGKey = None) -> SORL_Batch:
        """
        Reconstructs and returns a single Data object with zero-filled arrays.
        """
        # Convert the Pytree to a dictionary first.
        data_dict = self.data_structure._asdict()

        # Use a dictionary comprehension to build the zero arrays.
        reconstructed_arrays = {
            key: jnp.zeros(val[0], dtype=jnp.dtype(val[1]))
            for key, val in data_dict.items()
        }

        # Unpack the dictionary of arrays into the Data constructor.
        return SORL_Batch(**reconstructed_arrays)

    def __len__(self):
        # The dummy sampler conceptually contains one abstract batch.
        return 1