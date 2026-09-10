r"""Interpolation stencils: which pixels a sample reads, with what weights, and from where.

The type in this module carries no notion of pixelization. A landscape produces a stencil, a
sampler consumes one, and neither has to agree on anything else.
"""

from enum import IntEnum
from typing import NamedTuple, Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Integer

__all__ = [
    'Interpolation',
    'SkyPositions',
    'Stencil',
]


class Interpolation(IntEnum):
    """Which interpolation a sample gets, and therefore how many pixels it reads.

    The two are one choice, not two: nearest neighbour is the stencil of a single pixel, and
    bilinear the stencil of the four pixels around the sample. The enum value is the number of
    neighbours.

    Attributes:
        NEAREST: The single pixel the sample falls in.
        BILINEAR: The four pixels around the sample, weighted by the sub-pixel offset.
    """

    NEAREST = 1
    BILINEAR = 4


def _resolve(
    indices: Integer[Array, '*dims neighbors'], weights: Float[Array, '*dims neighbors']
) -> tuple[Integer[Array, '*dims neighbors'], Float[Array, '*dims neighbors']]:
    """Make an interpolation stencil safe to gather with, and normalize its weights.

    Neighbours outside the map (negative index) are sent to pixel 0 with their weight zeroed, so
    that the stencil can be gathered unconditionally, and the remaining weights are rescaled to sum
    to one, which keeps a partially covered sample unbiased. A sample with no neighbour left in the
    map has nothing to rescale: its weights stay at zero instead of being divided by zero, so it
    reads pixel 0 and contributes nothing.

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


class SkyPositions(NamedTuple):
    """Where the neighbours of a [`Stencil`][] sit on the sphere.

    The co-latitude is given as its cosine and its sine rather than as an angle, which is the form a
    HEALPix ring geometry produces and the form the spin-2 transport consumes.

    Attributes:
        z: Cosine of the neighbour co-latitude.
        sth: Sine of the neighbour co-latitude.
        phi: Neighbour longitude, in radians.
    """

    z: Float[Array, '*dims neighbors']
    sth: Float[Array, '*dims neighbors']
    phi: Float[Array, '*dims neighbors']

    def astype(self, dtype: DTypeLike) -> Self:
        """Return the positions cast to the given floating-point type."""
        return type(self)(*(component.astype(dtype) for component in self))


class Stencil(NamedTuple):
    r"""The pixels one sample reads, their weights, and where they sit on the sky.

    A stencil is *resolved*: its indices are safe to gather with unconditionally and its weights
    sum to one. Build one with [`Stencil.resolve`][] or [`Stencil.nearest`][] rather than by
    calling the constructor, which does not enforce that invariant.

    Nearest-neighbour sampling is the case of a single neighbour, not a different type: the
    trailing neighbour axis has length one and the weight is one.

    A stencil on a grid that is not the sphere has no [`SkyPositions`][] and carries `None`, which
    [`Stencil.scalar`][] builds; only a map with no polarisation can be sampled through one.

    Attributes:
        indices: Neighbour pixel indices into the raveled map, all in bounds.
        weights: Interpolation weights, one per neighbour, summing to one.
        positions: Where the neighbours sit on the sphere, or `None` off the sphere.
    """

    indices: Integer[Array, '*dims neighbors']
    weights: Float[Array, '*dims neighbors']
    positions: SkyPositions | None

    @property
    def n_neighbors(self) -> int:
        """Number of pixels each sample reads: one for nearest neighbour, four for bilinear."""
        return int(self.weights.shape[-1])

    def astype(self, dtype: DTypeLike) -> Self:
        """Return the stencil with its weights and positions cast to the given type.

        The indices are left untouched: they are integers, and casting them to a floating-point
        type would make them unusable as a gather index.

        Args:
            dtype: The floating-point type to cast to.

        Returns:
            The stencil, still resolved, in the requested type.
        """
        positions = None if self.positions is None else self.positions.astype(dtype)
        return type(self)(self.indices, self.weights.astype(dtype), positions)

    @classmethod
    def resolve(
        cls,
        indices: Integer[Array, '*dims neighbors'],
        weights: Float[Array, '*dims neighbors'],
        positions: SkyPositions | None,
    ) -> Self:
        """Build a stencil, sending out-of-map neighbours to a safe index and normalizing weights.

        Args:
            indices: Neighbour pixel indices, negative for neighbours outside the map.
            weights: Interpolation weights, one per neighbour, not necessarily normalized.
            positions: Where the neighbours sit on the sphere, or `None` off the sphere.

        Returns:
            The resolved stencil.
        """
        return cls(*_resolve(indices, weights), positions)

    @classmethod
    def nearest(
        cls,
        indices: Integer[Array, ' *dims'],
        theta_center: Float[Array, ' *dims'],
        phi_center: Float[Array, ' *dims'],
    ) -> Self:
        """Build the one-neighbour stencil of a nearest-neighbour sampler.

        Args:
            indices: Index of the pixel each sample falls in, negative outside the map.
            theta_center: Co-latitude of that pixel's center, in radians.
            phi_center: Longitude of that pixel's center, in radians.

        Returns:
            The resolved stencil, whose neighbour axis has length one.
        """
        return cls.resolve(
            indices[..., None],
            jnp.ones((*jnp.shape(indices), 1), jnp.result_type(theta_center)),
            SkyPositions(
                jnp.cos(theta_center)[..., None],
                jnp.sin(theta_center)[..., None],
                phi_center[..., None],
            ),
        )

    @classmethod
    def scalar(
        cls,
        indices: Integer[Array, '*dims neighbors'],
        weights: Float[Array, '*dims neighbors'],
    ) -> Self:
        """Build a stencil with no sky positions, for a grid that is not the sphere.

        The atmosphere screen is one: its pixels are a projection plane, so "where the neighbour
        sits on the sky" has no answer. Such a stencil can only sample a map with nothing to
        transport; anything reading its positions gets `None` rather than a plausible wrong number.

        Args:
            indices: Neighbour cell indices, negative for neighbours outside the grid.
            weights: Interpolation weights, one per neighbour, not necessarily normalized.

        Returns:
            The resolved stencil, with its positions set to `None`.
        """
        return cls.resolve(indices, weights, None)

    def reindexed(
        self, indices: Integer[Array, '*dims neighbors'], weights: Float[Array, '*dims neighbors']
    ) -> Self:
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
        return self.resolve(indices, weights, self.positions)
