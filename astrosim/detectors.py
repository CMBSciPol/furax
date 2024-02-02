from typing import Any

import jax
import numpy as np


class DetectorArray:
    # Z-axis is assumed to be the boresight of the telescope
    def __init__(self, x: Any, y: Any, z: Any):
        self.shape = np.broadcast(x, y, z).shape
        length = np.sqrt(x**2 + y**2 + z**2)
        coords = np.empty((3,) + self.shape)
        coords[0] = x
        coords[1] = y
        coords[2] = z
        coords /= length
        self.coords = jax.device_put(coords)

    def __len__(self) -> int:
        return int(np.prod(self.shape))
