r"""Spin-2 transported gather and scatter over an interpolation stencil."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from furax.obs.spin2._transport import spin2_cos_sin_zs
from furax.obs.stencil import Stencil
from furax.obs.stokes import Stokes

__all__ = [
    'transported_gather',
    'transported_scatter',
]


def transported_gather[S: Stokes](
    sky: S,
    stencil: Stencil,
    theta: Float[Array, ' *dims'],
    phi: Float[Array, ' *dims'],
) -> S:
    r"""Interpolate a flat sky map at world angles, transporting each neighbour's $(Q, U)$.

    Each neighbour's polarisation is carried into the meridian basis of the target direction before
    the weighted sum, so that the values being combined are components of one object. Without the
    transport the sum mixes bases and leaks $E$ into $B$. $I$ and $V$ are unaffected, and a
    [`StokesI`][] map is interpolated exactly as a scalar one.

    Args:
        sky: Sky map whose spatial axes are raveled, i.e. of shape ``(n_pixels,)`` per component.
        stencil: The pixels each sample reads, their weights and their positions.
        theta: Target co-latitude, in radians.
        phi: Target longitude, in radians.

    Returns:
        The interpolated Stokes values, of shape ``dims``.
    """
    gathered = type(sky).from_array(sky.data[..., stencil.indices])
    cos_2delta, sin_2delta = _transport_pair(gathered, stencil, theta, phi)
    rotated = gathered.rotate_qu(cos_2delta, sin_2delta)
    return type(sky).from_array(jnp.sum(rotated.data * stencil.weights, axis=-1))


def transported_scatter[S: Stokes](
    out: S,
    tod: S,
    stencil: Stencil,
    theta: Float[Array, ' *dims'],
    phi: Float[Array, ' *dims'],
) -> S:
    r"""Scatter-add samples into a flat sky map, adjoint to [`transported_gather`][].

    Each sample's polarisation is carried into the meridian basis of the neighbour it is deposited
    in, by the rotation inverse to the one [`transported_gather`][] applies to that neighbour.

    Args:
        out: Sky map to accumulate into, of the same shape as the map being sampled.
        tod: Samples to deposit, of shape ``dims``.
        stencil: The pixels each sample is deposited in, their weights and their positions.
        theta: Target co-latitude, in radians.
        phi: Target longitude, in radians.

    Returns:
        The accumulated sky map.
    """
    # Spread over the neighbour axis before rotating: the transport differs per neighbour, so each
    # copy of the sample turns by its own angle. The broadcast must be materialised, because
    # `rotate_qu` stacks the rotated Q, U rows back with the untouched I, V ones.
    spread = type(tod).from_array(
        jnp.broadcast_to(tod.data[..., None], (*tod.data.shape, stencil.n_neighbors))
    )
    cos_2delta, sin_2delta = _transport_pair(spread, stencil, theta, phi)
    # The transpose of rotate_qu(c, s) is rotate_qu(c, -s), which is what makes this the adjoint.
    rotated = spread.rotate_qu(cos_2delta, -sin_2delta)
    contrib = rotated.data * stencil.weights
    n_stokes = out.data.shape[0]
    accumulated = out.data.at[..., stencil.indices.ravel()].add(contrib.reshape(n_stokes, -1))
    return type(out).from_array(accumulated)


def _transport_pair(
    x: Stokes, stencil: Stencil, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
) -> tuple[Float[Array, '...'], Float[Array, '...']]:
    """Per-neighbour transport pair, or the identity pair when there is nothing to rotate.

    The identity pair is returned as a scalar rather than as an array of ones, so a map with no
    polarisation costs no memory here; it broadcasts against the neighbour axis unchanged.
    """
    if 'Q' not in x.stokes:
        return jnp.ones(()), jnp.zeros(())
    if stencil.z is None or stencil.sth is None or stencil.phi is None:
        raise ValueError(
            'the stencil carries no sky positions, so its Q and U cannot be transported; it '
            'describes a grid that is not the sphere and can only sample an intensity map'
        )
    return spin2_cos_sin_zs(
        stencil.z,
        stencil.sth,
        stencil.phi,
        jnp.cos(theta)[..., None],
        jnp.sin(theta)[..., None],
        phi[..., None],
    )
