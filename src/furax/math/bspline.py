from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int


def cubic_bspline(u: Float[Array, ' samp']) -> Float[Array, ' samp']:
    """Cubic cardinal B-spline, support [0,4)."""
    u = jnp.asarray(u)
    out = jnp.zeros_like(u)

    out = jnp.where((u >= 0) & (u < 1), (1 / 6) * u**3, out)
    out = jnp.where((u >= 1) & (u < 2), (1 / 6) * (-3 * u**3 + 12 * u**2 - 12 * u + 4), out)
    out = jnp.where((u >= 2) & (u < 3), (1 / 6) * (3 * u**3 - 24 * u**2 + 60 * u - 44), out)
    out = jnp.where((u >= 3) & (u < 4), (1 / 6) * (4 - u) ** 3, out)

    return out


def _spline_position(times: Float[Array, ' samp'], n_knots: int) -> Float[Array, ' samp']:
    """Sample positions on the knot grid: times rescaled so knot ``j`` sits at ``j``."""
    t_min = jnp.min(times)
    t_max = jnp.max(times)
    t = (times - t_min) / (t_max - t_min + 1e-12)  # to [0, 1]
    return t * (n_knots + 1)  # to [0, n_knots + 1], i.e. knot spacing 1


def spline_basis(times: Float[Array, ' samp'], n_knots: int) -> Float[Array, 'k samp']:
    """Dense cubic B-spline basis on ``K = n_knots + 2`` uniform knots.

    Returns:
        B: (K, N) basis matrix, ``B[j] = cubic_bspline(position - j + 1)``.
    """
    p = _spline_position(times, n_knots)
    K = n_knots + 2
    # knot j peaks at p = j (u = 2); evaluate all knots at once against every sample.
    u = p[None, :] - jnp.arange(K)[:, None] + 1.0
    return cubic_bspline(u)


def spline_window(
    times: Float[Array, ' samp'], n_knots: int
) -> tuple[Int[Array, ' samp'], Float[Array, 'samp 4']]:
    """Banded form of `spline_basis`: the 4 nonzero knots under each sample.

    A cubic B-spline reaches only 4 consecutive knots, so each sample's column of
    `spline_basis` has just 4 nonzero entries. This returns those directly, avoiding the
    mostly-zero ``(K, N)`` matrix (see `furax.mapmaking.templates.WindowedBasis`).

    Returns:
        offset: (N,) index of the first of the 4 knots, clamped to ``[0, K - 4]``.
        weights: (N, 4) the B-spline values at knots ``offset + 0..3``; weights of knots
            whose support misses the sample (at the time-range edges) are exactly zero.
    """
    p = _spline_position(times, n_knots)
    K = n_knots + 2
    offset = jnp.clip(jnp.floor(p).astype(jnp.int32) - 2, 0, K - 4)
    # knot offset+o contributes cubic_bspline(p - (offset+o) + 1); out-of-support -> 0.
    u = p[:, None] - (offset[:, None] + jnp.arange(4)) + 1.0
    return offset, cubic_bspline(u)
