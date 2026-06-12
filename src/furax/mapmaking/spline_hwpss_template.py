"""
Spline-based 4f HWP synchronous template for time-domain systematics modelling.

This module defines a linear template operator used to model and remove
time-varying systematic signals in detector time-ordered data (TOD),
in particular Half-Wave Plate (HWP) synchronous leakage at the 4f harmonic.

--------------------------------------------------------------------------------
PHYSICAL MODEL
--------------------------------------------------------------------------------

We model a detector signal as a modulated, time-varying systematic:

    d(t) = A(t) cos(4χ(t)) + B(t) sin(4χ(t))

where:
    - χ(t) is the HWP rotation angle (in radians)
    - 4χ(t) represents the dominant HWP synchronous harmonic (4f)
    - A(t), B(t) are slowly varying amplitude envelopes

Instead of assuming A(t), B(t) are constant or low-order polynomials,
we model them as smooth functions expanded in a cubic B-spline basis:

    A(t) = Σ_j a_j φ_j(t)
    B(t) = Σ_j b_j φ_j(t)

where:
    - φ_j(t) are cubic B-spline basis functions
    - a_j, b_j are unknown coefficients to be estimated

This leads to a linear model:

    d(t) = Σ_j a_j φ_j(t) cos(4χ(t)) + Σ_j b_j φ_j(t) sin(4χ(t))

which is linear in the coefficients and suitable for mapmaking and projection
frameworks.
"""

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
from .templates import TensorBasis
from .templates import PerDetectorTemplate


def cubic_bspline(u: Float[Array, 'n']) -> Float[Array, 'n']:
    """Cubic cardinal B-spline, support [0,4)."""
    u = jnp.asarray(u)
    out = jnp.zeros_like(u)

    out = jnp.where((u >= 0) & (u < 1), (1 / 6) * u**3, out)
    out = jnp.where((u >= 1) & (u < 2), (1 / 6) * (-3 * u**3 + 12 * u**2 - 12 * u + 4), out)
    out = jnp.where((u >= 2) & (u < 3), (1 / 6) * (3 * u**3 - 24 * u**2 + 60 * u - 44), out)
    out = jnp.where((u >= 3) & (u < 4), (1 / 6) * (4 - u) ** 3, out)

    return out


def spline_basis(
    times: Array,
    n_knots: int,
):
    """
    Returns:
        B: (K, N) spline basis matrix
    """
    t_min = jnp.min(times)
    t_max = jnp.max(times)
    t = (times - t_min) / (t_max - t_min + 1e-12)

    K = n_knots + 2
    spacing = 1.0 / (n_knots + 1)

    basis = []
    for j in range(K):
        center = (j + 1) * spacing
        u = (t - center) / spacing + 2.0
        basis.append(cubic_bspline(u))

    return jnp.stack(basis, axis=0)


def spline_4f_hwpss_basis(
    times: Array,
    hwp_angles: Array,
    n_knots: int,
):
    """
    Returns:
        B: (2K, N) basis matrix
            [phi_j cos(4χ), phi_j sin(4χ)]
    """
    phi = spline_basis(times, n_knots)  # (K, N)

    cos4 = jnp.cos(4.0 * hwp_angles)
    sin4 = jnp.sin(4.0 * hwp_angles)

    cols = []
    for j in range(phi.shape[0]):
        cols.append(phi[j] * cos4)
        cols.append(phi[j] * sin4)

    return jnp.stack(cols, axis=0)


def build_spline_4f_basis(
    times,
    hwp_angles,
    n_knots,
    dtype=jnp.float32,
):
    B = spline_4f_hwpss_basis(times, hwp_angles, n_knots)
    # B shape: (2K, N)

    return TensorBasis.create(B.astype(dtype))


def spline_4f_template(
    times,
    hwp_angles,
    n_dets,
    n_knots=20,
    dtype=jnp.float32,
):
    basis = build_spline_4f_basis(times, hwp_angles, n_knots, dtype)

    return PerDetectorTemplate.from_basis(
        basis,
        n_dets=n_dets,
        shared=True,
    )
