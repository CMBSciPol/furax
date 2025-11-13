from hashlib import sha1
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, UInt32


class DetectorArray:
    # Z-axis is assumed to be the boresight of the telescope
    def __init__(
        self,
        x: Float[Array, '*#dims'],
        y: Float[Array, '*#dims'],
        z: Float[Array | float, '*#dims'],
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

        # generate fake names for the detectors
        # TODO(simon): accept user-defined names
        widths = [len(str(s - 1)) for s in self.shape]
        indices = [[f'{i:0{width}}' for i in range(dim)] for dim, width in zip(self.shape, widths)]
        flat_names = ['DET_' + ''.join(combination) for combination in product(*indices)]
        self.names = np.array(flat_names).reshape(self.shape)

    def __len__(self) -> int:
        return int(np.prod(self.shape))

    def split_key(self, key: PRNGKeyArray) -> Shaped[PRNGKeyArray, ' _']:
        """Produces a new pseudo-random key for each detector."""
        fold = jax.numpy.vectorize(jax.random.fold_in, signature='(),()->()')
        subkeys: Shaped[PRNGKeyArray, ' ...'] = fold(key, self._ids())
        return subkeys

    def _ids(self) -> UInt32[Array, '...']:
        # vectorized hashing + converting to int + keeping only 7 bytes
        name_to_int = np.vectorize(lambda s: int(sha1(s.encode()).hexdigest(), 16) & 0xEFFFFFFF)
        # return detectors IDs as unsigned 32-bit integers
        ids: UInt32[Array, ' ...'] = jnp.uint32(name_to_int(self.names))
        return ids
