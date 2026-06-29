import flax
import jax
import jax.numpy as jnp
from typing import Any, NamedTuple
from sciql.core.data import Sampler
from pydantic import BaseModel

class SCBC_Data_Config(BaseModel):

    class Config:
        frozen = True  # Makes instances immutable and hashable.

    actor_p_curlabel: float | tuple[float] = 1.0
    actor_p_trajlabel: float | tuple[float] = 0.0
    actor_p_randomlabel: float | tuple[float] = 0.0
    actor_geom_lambda: float | tuple[float] = 0.99
    actor_dist_max: int | tuple[int] = 1000
    actor_geom_sample: bool | tuple[bool] = False

class SCBC_Data(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    labels: jnp.ndarray

class SCBC_Batch(NamedTuple):
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

    def sample_batch(self, batch_size: int, rng: jax.random.PRNGKey = None) -> SCBC_Batch:
        """
        Reconstructs and returns a single Data object with zero-filled arrays.
        """
        # THE FIX: Convert the Pytree to a dictionary first.
        data_dict = self.data_structure._asdict()

        # Use a dictionary comprehension to build the zero arrays.
        reconstructed_arrays = {
            key: jnp.zeros(val[0], dtype=jnp.dtype(val[1]))
            for key, val in data_dict.items()
        }

        # Unpack the dictionary of arrays into the Data constructor.
        return SCBC_Batch(**reconstructed_arrays)

    def __len__(self):
        # The dummy sampler conceptually contains one abstract batch.
        return 1