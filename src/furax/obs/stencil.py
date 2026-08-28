r"""Interpolation stencils: which pixels a sample reads, with what weights, and from where.

The type in this module carries no notion of pixelization. A landscape produces a stencil, a
sampler consumes one, and neither has to agree on anything else.
"""

from enum import IntEnum
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Integer

__all__ = [
    'Stencil',
    'StencilOrder',
    'resolve_stencil',
]


class StencilOrder(IntEnum):
    """How many pixels a sample reads, and therefore which interpolation it gets.

    The two are one choice, not two: nearest neighbour is the stencil of a single pixel, and
    bilinear the stencil of the four pixels around the sample. The enum value is the number of
    neighbours.

    Attributes:
        NEAREST: The single pixel the sample falls in.
        BILINEAR: The four pixels around the sample, weighted by the sub-pixel offset.
    """

    NEAREST = 1
    BILINEAR = 4


def resolve_stencil(
    indices: Integer[Array, '*dims neighbors'], weights: Float[Array, '*dims neighbors']
) -> tuple[Integer[Array, '*dims neighbors'], Float[Array, '*dims neighbors']]:
    """Make an interpolation stencil safe to gather with, and normalize its weights.

    Neighbours outside the map (negative index) are sent to pixel 0 with their weight zeroed, so
    that the stencil can be gathered unconditionally, and the remaining weights are rescaled to sum
    to one, which keeps a partially covered sample unbiased.

    Args:
        indices: Neighbour pixel indices, negative for neighbours outside the map.
        weights: Interpolation weights, one per neighbour.

    Returns:
        The in-bounds indices and the normalized weights.
    """
    # Every sampler must resolve a stencil the same way: the forward gather and the transposed
    # scatter are adjoint only if they normalize against identical weights.
    valid = indices >= 0
    indices = jnp.where(valid, indices, 0)
    weights = jnp.where(valid, weights, 0.0)
    weight_sum = weights.sum(axis=-1, keepdims=True)
    return indices, weights / jnp.where(weight_sum > 0, weight_sum, 1.0)


class Stencil(NamedTuple):
    r"""The pixels one sample reads, their weights, and where they sit on the sky.

    A stencil is *resolved*: its indices are safe to gather with unconditionally and its weights
    sum to one. Build one with [`Stencil.resolve`][] or [`Stencil.nearest`][] rather than by
    calling the constructor, which does not enforce that invariant.

    Nearest-neighbour sampling is the case of a single neighbour, not a different type: the
    trailing neighbour axis has length one and the weight is one.

    The neighbour co-latitude is given as its cosine and its sine rather than as an angle, which is
    the form a HEALPix ring geometry produces and the form the spin-2 transport consumes. A stencil
    on a grid that is not the sphere has no such positions and carries `None` for all three, which
    [`Stencil.scalar`][] builds; only a map with no polarisation can be sampled through one.

    Attributes:
        indices: Neighbour pixel indices into the raveled map, all in bounds.
        weights: Interpolation weights, one per neighbour, summing to one.
        z: Cosine of the neighbour co-latitude, or `None` off the sphere.
        sth: Sine of the neighbour co-latitude, or `None` off the sphere.
        phi: Neighbour longitude in radians, or `None` off the sphere.
    """

    indices: Integer[Array, '*dims neighbors']
    weights: Float[Array, '*dims neighbors']
    z: Float[Array, '*dims neighbors'] | None
    sth: Float[Array, '*dims neighbors'] | None
    phi: Float[Array, '*dims neighbors'] | None

    @property
    def n_neighbors(self) -> int:
        """Number of pixels each sample reads: one for nearest neighbour, four for bilinear."""
        return int(self.weights.shape[-1])

    @classmethod
    def resolve(
        cls,
        indices: Integer[Array, '*dims neighbors'],
        weights: Float[Array, '*dims neighbors'],
        z: Float[Array, '*dims neighbors'] | None,
        sth: Float[Array, '*dims neighbors'] | None,
        phi: Float[Array, '*dims neighbors'] | None,
        *,
        dtype: DTypeLike | None = None,
    ) -> 'Stencil':
        """Build a stencil, sending out-of-map neighbours to a safe index and normalizing weights.

        Args:
            indices: Neighbour pixel indices, negative for neighbours outside the map.
            weights: Interpolation weights, one per neighbour, not necessarily normalized.
            z: Cosine of the neighbour co-latitude, or `None` off the sphere.
            sth: Sine of the neighbour co-latitude, or `None` off the sphere.
            phi: Neighbour longitude in radians, or `None` off the sphere.
            dtype: If given, the floating-point type the weights and positions are cast to.

        Returns:
            The resolved stencil.
        """
        indices, weights = resolve_stencil(indices, weights)
        if dtype is not None:
            weights = weights.astype(dtype)
            z = None if z is None else z.astype(dtype)
            sth = None if sth is None else sth.astype(dtype)
            phi = None if phi is None else phi.astype(dtype)
        return cls(indices, weights, z, sth, phi)

    @classmethod
    def nearest(
        cls,
        indices: Integer[Array, ' *dims'],
        theta_center: Float[Array, ' *dims'],
        phi_center: Float[Array, ' *dims'],
        *,
        dtype: DTypeLike | None = None,
    ) -> 'Stencil':
        """Build the one-neighbour stencil of a nearest-neighbour sampler.

        Args:
            indices: Index of the pixel each sample falls in, negative outside the map.
            theta_center: Co-latitude of that pixel's center, in radians.
            phi_center: Longitude of that pixel's center, in radians.
            dtype: If given, the floating-point type the weights and positions are cast to.

        Returns:
            The resolved stencil, whose neighbour axis has length one.
        """
        return cls.resolve(
            indices[..., None],
            jnp.ones((*jnp.shape(indices), 1), dtype or jnp.result_type(theta_center)),
            jnp.cos(theta_center)[..., None],
            jnp.sin(theta_center)[..., None],
            phi_center[..., None],
            dtype=dtype,
        )

    @classmethod
    def scalar(
        cls,
        indices: Integer[Array, '*dims neighbors'],
        weights: Float[Array, '*dims neighbors'],
    ) -> 'Stencil':
        """Build a stencil with no sky positions, for a grid that is not the sphere.

        The atmosphere screen is one: its pixels are a projection plane, so "where the neighbour
        sits on the sky" has no answer. Such a stencil can only sample a map with nothing to
        transport; anything reading its positions gets `None` rather than a plausible wrong number.

        Args:
            indices: Neighbour cell indices, negative for neighbours outside the grid.
            weights: Interpolation weights, one per neighbour, not necessarily normalized.

        Returns:
            The resolved stencil, with `z`, `sth` and `phi` set to `None`.
        """
        return cls.resolve(indices, weights, None, None, None)

    def reindexed(
        self, indices: Integer[Array, '*dims neighbors'], weights: Float[Array, '*dims neighbors']
    ) -> 'Stencil':
        """Return the same neighbours addressed by new indices, with the weights re-resolved.

        For a landscape that re-numbers another one's pixels. The sky positions are unaffected by a
        re-numbering, so they are carried over as they are, but zeroing the weight of a neighbour
        the new numbering drops leaves the rest no longer summing to one, hence the re-resolution.

        Args:
            indices: Neighbour indices in the new numbering.
            weights: Weights in the new numbering, not necessarily normalized.

        Returns:
            The resolved stencil.
        """
        return Stencil.resolve(indices, weights, self.z, self.sth, self.phi)
