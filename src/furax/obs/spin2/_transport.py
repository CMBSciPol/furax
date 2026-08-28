r"""Spin-2 frame rotation between two directions on the sphere."""

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = [
    'spin2_cos_sin',
    'spin2_cos_sin_zs',
]


def spin2_cos_sin_zs(
    z_n: Float[Array, '...'],
    s_n: Float[Array, '...'],
    phi_n: Float[Array, '...'],
    z_x: Float[Array, '...'],
    s_x: Float[Array, '...'],
    phi_x: Float[Array, '...'],
) -> tuple[Float[Array, '...'], Float[Array, '...']]:
    r"""Spin-2 transport pair between two directions given as $(\cos\theta, \sin\theta, \varphi)$.

    Same as [`spin2_cos_sin`][], but takes the cosine and sine of each co-latitude instead of the
    co-latitude itself. Prefer this form when they are already available, as they are for HEALPix
    pixel centres.

    Args:
        z_n: Cosine of the neighbour co-latitude.
        s_n: Sine of the neighbour co-latitude.
        phi_n: Neighbour longitude, in radians.
        z_x: Cosine of the target co-latitude.
        s_x: Sine of the target co-latitude.
        phi_x: Target longitude, in radians.

    Returns:
        The pair $(\cos 2\delta,\, -\sin 2\delta)$, as taken by [`Stokes.rotate_qu`][]. Broadcasts
        over the six inputs.
    """
    # Haversine formulation. The textbook form -- two atan2 bearings -- is algebraically identical
    # but cancels catastrophically at the sub-pixel separations this is used at, losing about four
    # digits in float32. Do not "simplify" this back to atan2.
    dphi = phi_x - phi_n
    ch = jnp.cos(0.5 * dphi)
    sh = jnp.sin(0.5 * dphi)
    sin_dphi = 2.0 * ch * sh
    cos_dphi = 1.0 - 2.0 * sh * sh
    sh2 = sh * sh

    dz = z_x - z_n
    ds = s_x - s_n
    st2 = 0.25 * (dz * dz + ds * ds)  # sin^2(dtheta / 2)
    prod_s = s_n * s_x
    hav = st2 + prod_s * sh2  # haversine of the angular separation
    sin2_dtheta = 4.0 * st2 * (1.0 - st2)

    num = 2.0 * sin_dphi * (z_x + z_n) * hav
    den = (
        sin2_dtheta * cos_dphi - 4.0 * prod_s * z_x * z_n * sh2 * sh2 + prod_s * sin_dphi * sin_dphi
    )

    # tan(delta) = num / den. The double-angle identities recover the pair from the tangent without
    # ever forming the angle, which is where the precision at small separation comes from.
    # norm vanishes only when the two directions coincide; the doubled where keeps the gradient of
    # the dead branch finite.
    norm = num * num + den * den
    inv = 1.0 / jnp.where(norm > 0, norm, 1.0)
    cos_2delta = jnp.where(norm > 0, (den * den - num * num) * inv, 1.0)
    sin_2delta = jnp.where(norm > 0, 2.0 * num * den * inv, 0.0)
    return cos_2delta, sin_2delta


def spin2_cos_sin(
    theta_n: Float[Array, '...'],
    phi_n: Float[Array, '...'],
    theta_x: Float[Array, '...'],
    phi_x: Float[Array, '...'],
) -> tuple[Float[Array, '...'], Float[Array, '...']]:
    r"""Spin-2 transport pair between two directions, in the form taken by `rotate_qu`.

    A polarisation $P = Q + iU$ stored at a neighbour direction $\hat n$ is expressed in the local
    meridian basis at $\hat n$. Carrying it to the basis at a target direction $\hat x$ turns it by
    the transport angle $\delta$, as $P \to P e^{+2i\delta}$.

    The returned pair is $(\cos 2\delta,\, -\sin 2\delta)$: [`Stokes.rotate_qu`][] applies
    $P \to P e^{-2ia}$ to a pair $(\cos 2a,\, \sin 2a)$, so passing this pair there directly, with
    no sign flip, performs the transport. It must not be passed to [`rotate_qu_cs`][], which expects
    a single-angle pair and doubles it internally.

    The sign convention is that of HEALPix/COSMO maps. An IAU-convention map has $U$ flipped and
    would run the transport backwards.

    Coincident directions give $(1, 0)$ exactly. Directions sharing a meridian give it to
    round-off.

    Args:
        theta_n: Neighbour co-latitude, in radians.
        phi_n: Neighbour longitude, in radians.
        theta_x: Target co-latitude, in radians.
        phi_x: Target longitude, in radians.

    Returns:
        The pair $(\cos 2\delta,\, -\sin 2\delta)$, as taken by [`Stokes.rotate_qu`][]. Broadcasts
        over the four inputs.

    Examples:
        Two directions on the same meridian need no rotation.

        >>> import jax.numpy as jnp
        >>> cos_2delta, sin_2delta = spin2_cos_sin(
        ...     jnp.asarray(0.3), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(0.0)
        ... )
        >>> round(float(cos_2delta), 12), round(float(sin_2delta), 12)
        (1.0, 0.0)
    """
    return spin2_cos_sin_zs(
        jnp.cos(theta_n), jnp.sin(theta_n), phi_n, jnp.cos(theta_x), jnp.sin(theta_x), phi_x
    )
