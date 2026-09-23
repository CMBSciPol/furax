import dataclasses
from dataclasses import dataclass, field
from typing import Self

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jaxtyping import Array, Float, Int

from furax.obs.landscapes import StokesLandscape
from furax.obs.stencil import Interpolation, Stencil
from furax.obs.stokes import Stokes

__all__ = [
    'SamplingKernel',
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SamplingKernel:
    r"""The integration kernel for one sample.

    A sample reads the map at one or more directions and returns their weighted sum:

    $$d = \sum_k w_k \, m(\hat{n}_k),$$

    where $\hat{n}_k$, a read direction, is the line of sight of the sample composed with
    offset $k$. Without offsets, the only read direction is the line of sight, with unit
    weight. The offsets model an integration within a sample: a finite time integration, a
    pixel window, or a beam.

    Each value $m(\hat{n}_k)$ is then read from the pixelized map with the kernel's interpolation,
    independently of the offsets: with bilinear interpolation, every direction reads its four
    nearest pixels, so a kernel of $K$ offsets reads up to $4K$ pixels per sample.

    Build one with [`SamplingKernel.create`][], which validates the offsets and weights against
    the map and the number of detectors; the constructor does not.

    Attributes:
        interpolation: How each direction is read from the map.
        offsets: Rotations from the line of sight to each read direction, shape (n_offsets,)
            for the same offsets on every detector, or (n_detectors, n_offsets). `None` to read
            the line of sight alone.
        weights: The non-negative weight of each offset, shape (n_offsets,), shared by every
            Stokes component, or a [`Stokes`][] of the map's components, each of shape
            (n_offsets,). `None` if `offsets` is `None`.
    """

    interpolation: Interpolation = field(default=Interpolation.NEAREST, metadata={'static': True})
    offsets: Quaternion | None = None
    weights: Float[Array, ' n_offsets'] | Stokes | None = None

    @classmethod
    def create(
        cls,
        landscape: StokesLandscape,
        n_detectors: int,
        *,
        interpolation: Interpolation = Interpolation.NEAREST,
        offsets: Quaternion | None = None,
        weights: Float[Array, ' n_offsets'] | Stokes | None = None,
    ) -> Self:
        """Build a kernel, checking the offsets and weights against the map and detectors.

        The weights are cast to the map's dtype. They are normalized to sum to one, per Stokes
        component, over the offsets whose pixels are in the map: a sample partly off a partial-sky
        map is renormalized to the part in view, like a partly covered bilinear sample.
        """
        if (offsets is None) != (weights is None):
            raise ValueError('offsets and their weights must be given together')
        if offsets is None:
            return cls(interpolation)
        assert weights is not None
        n_offsets = offsets.shape[-1]
        if offsets.shape not in {(n_offsets,), (n_detectors, n_offsets)}:
            raise ValueError(
                f'offsets has shape {offsets.shape}, expected ({n_offsets},) or '
                f'({n_detectors}, {n_offsets}) for {n_detectors} detectors'
            )
        return cls(interpolation, offsets, _checked_weights(weights, landscape, n_offsets))

    def integrate(self, stencil: Stencil) -> Stencil:
        """Fold stencils of shape (..., n_offsets, neighbors) into (..., n_offsets * neighbors)."""
        assert self.weights is not None
        weights = self.weights.data if isinstance(self.weights, Stokes) else self.weights
        return stencil.integrated(weights)

    def offsets_for(self, idet: Int[Array, ' batch']) -> Quaternion | None:
        """Shape (batch, n_offsets), with shared offsets broadcast to every detector."""
        if self.offsets is None:
            return None
        if len(self.offsets.shape) == 1:
            # shared by every detector
            shape = (*idet.shape, *self.offsets.wxyz.shape)
            return Quaternion.from_array(jnp.broadcast_to(self.offsets.wxyz, shape))
        return self.offsets[idet]

    def intensity_only(self) -> Self:
        """Reduce per-Stokes weights to those of I, or to their mean when the map has no I."""
        weights = self.weights
        if not isinstance(weights, Stokes):
            return self
        reduced = weights.i if 'I' in weights.stokes else weights.data.mean(0)
        return dataclasses.replace(self, weights=reduced)


def _checked_weights(
    weights: Float[Array, ' n_offsets'] | Stokes, landscape: StokesLandscape, n_offsets: int
) -> Float[Array, ' n_offsets'] | Stokes:
    """The offset weights in the landscape's dtype, after checking they fit the offsets and map."""
    if isinstance(weights, Stokes):
        if weights.stokes != landscape.stokes:
            raise ValueError(
                f'offset weights have Stokes components {weights.stokes!r}, expected those of '
                f'the landscape, {landscape.stokes!r}'
            )
        if weights.shape != (n_offsets,):
            raise ValueError(
                f'offset weights have shape {weights.shape} per component, expected '
                f'({n_offsets},) for {n_offsets} offsets'
            )
        return type(weights).from_array(jnp.asarray(weights.data, dtype=landscape.dtype))
    weights = jnp.asarray(weights, dtype=landscape.dtype)
    if weights.shape != (n_offsets,):
        hint = (
            ', or give one set per Stokes component as a Stokes, '
            'e.g. StokesIQU(i=..., q=..., u=...)'
            if weights.ndim == 2
            else ''
        )
        raise ValueError(
            f'offset weights have shape {weights.shape}, expected ({n_offsets},) for '
            f'{n_offsets} offsets{hint}'
        )
    return weights
