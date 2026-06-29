import flax
import jax
import jax.numpy as jnp
from typing import Any, NamedTuple
from sciql.core.data import Sampler

class CBC_Data(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    labels: jnp.ndarray

class CBC_Batch(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    labels: jnp.ndarray

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

    def sample_batch(self, batch_size: int, rng: jax.random.PRNGKey = None) -> CBC_Batch:
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
        return CBC_Batch(**reconstructed_arrays)

    def __len__(self):
        # The dummy sampler conceptually contains one abstract batch.
        return 1