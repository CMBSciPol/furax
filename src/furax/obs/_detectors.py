from hashlib import sha1
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, PRNGKeyArray, Shaped, UInt32


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
        fold = jnp.vectorize(jax.random.fold_in, signature='(),()->()')
        uids = jnp.asarray(names_to_uids(self.names))
        return fold(key, uids)  # type: ignore[no-any-return]


def names_to_uids(names: Shaped[np.ndarray, '*#dims']) -> UInt32[np.ndarray, '*#dims']:
    """Converts names to unsigned 32-bit integers using hashing.

    This is typically used to generate deterministic uids for detectors based on their names.
    """
    # vectorized hashing + converting to int + keeping only 7 bytes
    name_to_int = np.vectorize(lambda s: int(sha1(s.encode()).hexdigest(), 16) & 0xEFFFFFFF)
    return name_to_int(names).astype(np.uint32)  # type: ignore[no-any-return]
