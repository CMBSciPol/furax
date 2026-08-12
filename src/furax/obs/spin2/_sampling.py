r"""Spin-2 transported gather and scatter over an interpolation stencil."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from furax.obs.landscapes import InterpCenters
from furax.obs.spin2._transport import spin2_cos_sin_zs
from furax.obs.stokes import Stokes

__all__ = [
    'transported_gather',
    'transported_scatter',
]


def transported_gather[StokesT: Stokes](
    sky: StokesT,
    indices: Integer[Array, '*dims neighbors'],
    weights: Float[Array, '*dims neighbors'],
    centers: InterpCenters,
    theta: Float[Array, ' *dims'],
    phi: Float[Array, ' *dims'],
) -> StokesT:
    r"""Interpolate a flat sky map at world angles, transporting each neighbour's $(Q, U)$.

    Each neighbour's polarisation is carried into the meridian basis of the target direction before
    the weighted sum, so that the values being combined are components of one object. Without the
    transport the sum mixes bases and leaks $E$ into $B$. $I$ and $V$ are unaffected, and a
    [`StokesI`][] map is interpolated exactly as a scalar one.

    Args:
        sky: Sky map whose spatial axes are raveled, i.e. of shape ``(n_pixels,)`` per component.
        indices: Neighbour pixel indices, negative for neighbours outside the map.
        weights: Interpolation weights, one per neighbour.
        centers: World positions of the neighbours.
        theta: Target co-latitude, in radians.
        phi: Target longitude, in radians.

    Returns:
        The interpolated Stokes values, of shape ``dims``.
    """
    indices, unit_weights = _resolve_stencil(indices, weights)
    gathered = type(sky).from_array(sky.data[..., indices])
    cos_2delta, sin_2delta = _transport_pair(gathered, centers, theta, phi)
    rotated = gathered.rotate_qu(cos_2delta, sin_2delta)
    return type(sky).from_array(jnp.sum(rotated.data * unit_weights, axis=-1))


def transported_scatter[StokesT: Stokes](
    out: StokesT,
    tod: StokesT,
    indices: Integer[Array, '*dims neighbors'],
    weights: Float[Array, '*dims neighbors'],
    centers: InterpCenters,
    theta: Float[Array, ' *dims'],
    phi: Float[Array, ' *dims'],
) -> StokesT:
    r"""Scatter-add samples into a flat sky map, adjoint to [`transported_gather`][].

    Each sample's polarisation is carried into the meridian basis of the neighbour it is deposited
    in, by the rotation inverse to the one [`transported_gather`][] applies to that neighbour.

    Args:
        out: Sky map to accumulate into, of the same shape as the map being sampled.
        tod: Samples to deposit, of shape ``dims``.
        indices: Neighbour pixel indices, negative for neighbours outside the map.
        weights: Interpolation weights, one per neighbour.
        centers: World positions of the neighbours.
        theta: Target co-latitude, in radians.
        phi: Target longitude, in radians.

    Returns:
        The accumulated sky map.
    """
    indices, unit_weights = _resolve_stencil(indices, weights)
    # Spread over the neighbour axis before rotating: the transport differs per neighbour, so each
    # copy of the sample turns by its own angle. The broadcast must be materialised, because
    # `rotate_qu` stacks the rotated Q, U rows back with the untouched I, V ones.
    n_neighbors = unit_weights.shape[-1]
    spread = type(tod).from_array(
        jnp.broadcast_to(tod.data[..., None], (*tod.data.shape, n_neighbors))
    )
    cos_2delta, sin_2delta = _transport_pair(spread, centers, theta, phi)
    # The transpose of rotate_qu(c, s) is rotate_qu(c, -s), which is what makes this the adjoint.
    rotated = spread.rotate_qu(cos_2delta, -sin_2delta)
    contrib = rotated.data * unit_weights
    n_stokes = out.data.shape[0]
    accumulated = out.data.at[..., indices.ravel()].add(contrib.reshape(n_stokes, -1))
    return type(out).from_array(accumulated)


def _resolve_stencil(
    indices: Integer[Array, '*dims neighbors'], weights: Float[Array, '*dims neighbors']
) -> tuple[Integer[Array, '*dims neighbors'], Float[Array, '*dims neighbors']]:
    """Send out-of-bounds neighbours to pixel 0 with zero weight, and renormalize.

    Renormalizing keeps partially covered samples unbiased. It must happen identically in the gather
    and in the scatter, or the two stop being adjoint.
    """
    valid = indices >= 0
    indices = jnp.where(valid, indices, 0)
    weights = jnp.where(valid, weights, 0.0)
    weight_sum = weights.sum(axis=-1, keepdims=True)
    return indices, weights / jnp.where(weight_sum > 0, weight_sum, 1.0)


def _transport_pair(
    x: Stokes, centers: InterpCenters, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
) -> tuple[Float[Array, '*dims neighbors'], Float[Array, '*dims neighbors']]:
    """Per-neighbour transport pair, or the identity pair when there is nothing to rotate."""
    if 'Q' not in x.stokes:
        return jnp.ones(()), jnp.zeros(())
    return spin2_cos_sin_zs(
        centers.z,
        centers.sth,
        centers.phi,
        jnp.cos(theta)[..., None],
        jnp.sin(theta)[..., None],
        phi[..., None],
    )
