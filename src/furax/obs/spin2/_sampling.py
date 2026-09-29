r"""Spin-2 transported gather and scatter over an interpolation stencil."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from furax.math.coords import ZSPhi
from furax.obs.operators._qu_rotations import Spin2Rotation
from furax.obs.spin2._transport import spin2_cos_sin_zs
from furax.obs.stencil import Stencil
from furax.obs.stokes import Stokes

__all__ = [
    'rotated_gather',
    'rotated_scatter',
    'transport_rotation',
    'transported_gather',
    'transported_scatter',
]


def transported_gather[S: Stokes](
    sky: S,
    stencil: Stencil,
    theta: Float[Array, ' *dims'],
    phi: Float[Array, ' *dims'],
    *,
    rotation: Spin2Rotation | None = None,
) -> S:
    r"""Interpolate a flat sky map at world angles, transporting each neighbour's $(Q, U)$.

    Each neighbour's polarisation is carried into the meridian basis of the target direction before
    the weighted sum, so that the values being combined are components of one object. Without the
    transport the sum mixes bases and leaks $E$ into $B$. $I$ and $V$ are unaffected, and a
    [`StokesI`][] map is interpolated exactly as a scalar one.

    A `rotation` by an angle $\psi$ turns every neighbour further, from the meridian basis into a
    basis rotated by $\psi$ from it, such as a detector's, before the weighted sum. Stencil weights
    with a leading Stokes axis weigh each component on its own, in the basis reached: the meridian
    basis of the target direction, or the rotated basis if a `rotation` is given. The two orders
    differ once the weights differ between $Q$ and $U$, since such a weighting does not commute
    with a rotation.

    Args:
        sky: Sky map whose spatial axes are raveled, i.e. of shape ``(n_pixels,)`` per component.
        stencil: The pixels each sample reads, their weights and their positions.
        theta: Target co-latitude, in radians.
        phi: Target longitude, in radians.
        rotation: The rotation by $\psi$ applied after the transport and before the weights, or
            `None` to stay in the meridian basis of the target direction.

    Returns:
        The interpolated Stokes values, of shape ``dims``.
    """
    if 'Q' not in sky.stokes:
        return rotated_gather(sky, stencil, None)
    target = ZSPhi.from_angles(theta, phi)
    return rotated_gather(sky, stencil, transport_rotation(stencil, target, rotation))


def transported_scatter[S: Stokes](
    out: S,
    tod: S,
    stencil: Stencil,
    theta: Float[Array, ' *dims'],
    phi: Float[Array, ' *dims'],
    *,
    rotation: Spin2Rotation | None = None,
) -> S:
    r"""Scatter-add samples into a flat sky map, adjoint to [`transported_gather`][].

    Each sample's polarisation is carried into the meridian basis of the neighbour it is deposited
    in, by the rotation inverse to the one [`transported_gather`][] applies to that neighbour,
    including the same `rotation`.

    Args:
        out: Sky map to accumulate into, of the same shape as the map being sampled.
        tod: Samples to deposit, of shape ``dims``.
        stencil: The pixels each sample is deposited in, their weights and their positions.
        theta: Target co-latitude, in radians.
        phi: Target longitude, in radians.
        rotation: The rotation given to [`transported_gather`][], or `None`.

    Returns:
        The accumulated sky map.
    """
    if 'Q' not in tod.stokes:
        return rotated_scatter(out, tod, stencil, None)
    target = ZSPhi.from_angles(theta, phi)
    return rotated_scatter(out, tod, stencil, transport_rotation(stencil, target, rotation))


def rotated_gather[S: Stokes](sky: S, stencil: Stencil, rotation: Spin2Rotation | None) -> S:
    r"""Read a flat sky map through a stencil, rotating each neighbour's $(Q, U)$ by its own angle.

    Sample $s$ is $\sum_n w_{sn} R(\alpha_{sn}) m_{p_{sn}}$: each neighbour $n$ of the stencil is
    read at pixel $p_{sn}$, its $(Q, U)$ rotated by $\alpha_{sn}$, then weighted. $I$ and $V$ are
    unaffected. [`transported_gather`][] is the case where $\alpha$ is the parallel transport
    to the sampled direction, computed by [`transport_rotation`][].

    Args:
        sky: Sky map whose spatial axes are raveled, i.e. of shape `(n_pixels,)` per component.
        stencil: The pixels each sample reads and their weights, optionally one row per Stokes
            component. Its positions are not used.
        rotation: The rotation by $\alpha$ of each neighbour, broadcastable to the stencil
            indices, or `None` to leave the neighbours unrotated.

    Returns:
        The sampled Stokes values, of the shape of the stencil minus its neighbour axis.
    """
    if stencil.n_neighbors == 1:
        gathered = type(sky).from_array(sky.data[..., stencil.indices])
        if rotation is not None:
            gathered = gathered.rotate_qu(*rotation)
        return type(sky).from_array(jnp.sum(gathered.data * stencil.weights, axis=-1))
    # With several neighbours, each component is gathered, rotated and summed over them before
    # the rows are stacked, so that XLA fuses the gathers into the sums instead of materialising
    # every neighbour of every component, which `Stokes.rotate_qu` on the gathered map would
    # force. With a single neighbour there is nothing to sum, and one gather of all the rows is
    # faster.
    rows = [sky.data[i][stencil.indices] for i in range(len(sky.stokes))]
    qi = sky.stokes.find('Q')
    if rotation is not None and qi >= 0:
        cos_2a, sin_2a = rotation
        q, u = rows[qi], rows[qi + 1]
        rows[qi], rows[qi + 1] = q * cos_2a + u * sin_2a, -q * sin_2a + u * cos_2a
    per_stokes = stencil.weights.ndim > stencil.indices.ndim
    sums = [
        jnp.sum(row * (stencil.weights[i] if per_stokes else stencil.weights), axis=-1)
        for i, row in enumerate(rows)
    ]
    return type(sky).from_array(jnp.stack(sums))


def rotated_scatter[S: Stokes](
    out: S, tod: S, stencil: Stencil, rotation: Spin2Rotation | None
) -> S:
    """Scatter-add samples into a flat sky map, adjoint to [`rotated_gather`][].

    Args:
        out: Sky map to accumulate into, of the same shape as the map being sampled.
        tod: Samples to deposit, of the shape of the stencil minus its neighbour axis.
        stencil: The pixels each sample is deposited in and their weights.
        rotation: The rotation given to [`rotated_gather`][], or `None`.

    Returns:
        The accumulated sky map.
    """
    # Spread over the neighbour axis before rotating: the rotation differs per neighbour, so each
    # copy of the sample turns by its own angle. Weighting comes first, mirroring the gather in
    # reverse: the gather weights *after* rotating, so the adjoint weights in that basis *before*
    # rotating back. The order matters once the weights differ between Q and U, since a
    # per-component weight does not commute with the rotation. The spread is materialised (not a
    # lazy broadcast) because `rotate_qu` stacks the rotated Q, U rows back with the untouched I,
    # V ones.
    spread = type(tod).from_array(
        jnp.broadcast_to(tod.data[..., None], (*tod.data.shape, stencil.n_neighbors))
        * stencil.weights
    )
    if rotation is not None:
        # the transpose of a rotation is its inverse, which is what makes this the adjoint
        spread = spread.rotate_qu(*rotation.inverse())
    n_stokes = out.data.shape[0]
    contrib = spread.data.reshape(n_stokes, -1)
    accumulated = out.data.at[..., stencil.indices.ravel()].add(contrib)
    return type(out).from_array(accumulated)


def transport_rotation(
    stencil: Stencil,
    target: ZSPhi,
    rotation: Spin2Rotation | None = None,
) -> Spin2Rotation:
    r"""The rotation of each neighbour's $(Q, U)$ into the basis of the sampled direction.

    The parallel transport $\delta$ from each neighbour's meridian basis to that of the target
    direction, followed by a rotation $\psi$ shared by the neighbours of a sample, if given.

    Args:
        stencil: The pixels each sample reads, with their positions.
        target: The direction each sample is transported to, of shape `dims` (no neighbour
            axis).
        rotation: The rotation by $\psi$ after the transport, or `None`.

    Returns:
        The rotation by $\alpha = \delta + \psi$ of each neighbour.
    """
    if stencil.positions is None:
        raise ValueError(
            'the stencil carries no sky positions, so its Q and U cannot be transported; it '
            'describes a grid that is not the sphere and can only sample an intensity map'
        )
    transport = spin2_cos_sin_zs(
        *stencil.positions, *(component[..., None] for component in target)
    )
    if rotation is None:
        return transport
    # psi is shared by the neighbours of a sample
    return transport.compose(rotation[..., None])
