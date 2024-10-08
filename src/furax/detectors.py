from functools import reduce
from hashlib import sha1

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, UInt32


class DetectorArray:
    # Z-axis is assumed to be the boresight of the telescope
    def __init__(
        self,
        x: Float[np.ndarray, '*#dims'],  # type: ignore[type-arg]
        y: Float[np.ndarray, '*#dims'],  # type: ignore[type-arg]
        z: Float[np.ndarray | float, '*#dims'],  # type: ignore[type-arg]
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
        det_indices = (_.astype(np.str_) for _ in np.indices(self.shape, sparse=True))
        names = reduce(lambda a, b: a + '_' + b, det_indices, 'FAKE_DET')
        self.names = names

    def __len__(self) -> int:
        return int(np.prod(self.shape))

    def _ids(self) -> UInt32[Array, '...']:
        # vectorized hashing + converting to int + keeping only 7 bytes
        name_to_int = np.vectorize(lambda s: int(sha1(s.encode()).hexdigest(), 16) & 0xEFFFFFFF)
        # return detectors IDs as unsigned 32-bit integers
        ids: UInt32[Array, '...'] = jnp.uint32(name_to_int(self.names))
        return ids

    def split_key(self, key: PRNGKeyArray) -> Shaped[PRNGKeyArray, ' _']:
        """Folds the detector names in a random key to generate a key array."""
        fold = jax.numpy.vectorize(jax.random.fold_in, signature='(),()->()')
        subkeys: Shaped[PRNGKeyArray, '...'] = fold(key, self._ids())
        return subkeys


class FakeDetectorArray(DetectorArray):
    def __init__(self, num: int | tuple[int, ...]) -> None:
        super().__init__(np.zeros(num), np.zeros(num), 1.0)
