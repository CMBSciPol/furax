import jax
import numpy as np
from jaxtyping import Float


class DetectorArray:
    # Z-axis is assumed to be the boresight of the telescope
    def __init__(
        self,
        x: Float[np.ndarray, '*#dims'],
        y: Float[np.ndarray, '*#dims'],
        z: Float[np.ndarray | float, '*#dims'],
    ) -> None:
        self.shape = np.broadcast(
            x, y, z
        ).shape  # FIXME: check jax broadcast so that we can accept Arrays
        length = np.sqrt(x**2 + y**2 + z**2)
        coords = np.empty((3,) + self.shape)
        coords[0] = x
        coords[1] = y
        coords[2] = z
        coords /= length
        self.coords = jax.device_put(coords)

    def __len__(self) -> int:
        return int(np.prod(self.shape))
