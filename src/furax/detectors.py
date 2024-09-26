from hashlib import sha1

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, PRNGKeyArray, Shaped


class DetectorArray:
    # Z-axis is assumed to be the boresight of the telescope
    def __init__(
        self,
        x: Float[np.ndarray, '*#dims'],  # type: ignore[type-arg]
        y: Float[np.ndarray, '*#dims'],  # type: ignore[type-arg]
        z: Float[np.ndarray | float, '*#dims'],  # type: ignore[type-arg]
        names: list[str] | None = None,
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

        # how many detectors there are
        n_detectors = 1 if len(self.shape) == 1 else self.shape[-2]
        if names is None:
            # if not provided, create some detector names
            names = [f'DET_{i}' for i in range(n_detectors)]
        assert len(names) == n_detectors
        self.names = names

    def __len__(self) -> int:
        return int(np.prod(self.shape))

    def split_key(self, key: PRNGKeyArray) -> Shaped[PRNGKeyArray, ' _']:
        """Folds the detector names in a random key to generate a key array."""
        fold = jax.numpy.vectorize(jax.random.fold_in, signature='(),()->()')
        det_hashes = [int(sha1(name.encode()).hexdigest(), 16) for name in self.names]
        data = jnp.uint32([dh & 0xEFFFFFFF for dh in det_hashes])
        subkeys: Shaped[PRNGKeyArray, '...'] = fold(key, data)
        return subkeys


class FakeDetectorArray(DetectorArray):
    def __init__(self, shape: int | tuple[int, ...], names: list[str] | None = None) -> None:
        super().__init__(np.zeros(shape), np.zeros(shape), 1.0, names)
