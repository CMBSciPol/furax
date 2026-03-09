import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, Float


class DetectorArray:
    # Z-axis is assumed to be the boresight of the telescope
    def __init__(
        self,
        x: Float[ArrayLike, '*#dims'],
        y: Float[ArrayLike, '*#dims'],
        z: Float[ArrayLike, '*#dims'],
    ) -> None:
        x, y, z = jnp.broadcast_arrays(x, y, z)
        self.shape = x.shape
        length = jnp.sqrt(x**2 + y**2 + z**2)
        self.coords = jnp.stack((x, y, z), axis=-1) / length

    def __len__(self) -> int:
        return int(np.prod(self.shape))
