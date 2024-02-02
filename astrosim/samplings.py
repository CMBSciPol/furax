from dataclasses import dataclass

import jax
import jax_healpy as jhp
import numpy as np
from jaxtyping import Array, Float
from scipy.stats.sampling import DiscreteAliasUrn


@dataclass(frozen=True)
class Sampling:
    theta: Float[Array, '...']
    phi: Float[Array, '...']
    pa: Float[Array, '...']

    def __len__(self) -> int:
        return np.broadcast(self.theta, self.phi, self.pa).size


def create_random_sampling(
    hit_map, nsampling: int, random_generator: np.random.Generator
) -> Sampling:
    npixel = hit_map.size
    nside = jhp.npix2nside(npixel)
    rng = DiscreteAliasUrn(hit_map, random_state=random_generator)
    ipixels = rng.rvs(size=nsampling)

    theta, phi = jhp.pix2ang(nside, ipixels)
    pa = random_generator.uniform(0, 2 * np.pi, nsampling)

    return Sampling(jax.device_put(theta), jax.device_put(phi), jax.device_put(pa))
