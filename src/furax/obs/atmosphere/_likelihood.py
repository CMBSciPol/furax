import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Scalar

from furax import AbstractLinearOperator, DiagonalOperator
from furax.obs.stokes import StokesI
from furax.tree import dot, ones_like

__all__ = [
    'profile_neg_log_likelihood',
]


def profile_neg_log_likelihood(
    pointing_op: AbstractLinearOperator,
    d: StokesI,
    noise_cov_inv: AbstractLinearOperator,
    *,
    solver: lx.AbstractLinearSolver | None = None,
) -> Scalar:
    r"""Negative profile log-likelihood for atmosphere pointing parameters.

    Assumes a data model `d = P(v) m + n` where P(v) is the atmosphere pointing
    operator parameterised by wind velocity `v` (or any other pointing parameters).

    Computes the "spectral" (profile) negative log-likelihood given (up to a constant) by
    `-2 log L(v) = (P.T N.I d).T (P.T N.I P).I (P.T N.I d)` for use with a minimiser.

    Examples:
        >>> import lineax as lx
        >>> def loss(wind_velocity):
        ...     P = AtmospherePointingOperator.from_wind(
        ...         landscape, qbore, qdet, wind_velocity, times, interpolate=True
        ...     )
        ...     return profile_neg_log_likelihood(P, d_obs, N_inv)
        >>> grad = jax.grad(loss)(wind_velocity_init)

    Args:
        pointing_op: Atmosphere pointing operator `P(v)` for current parameters.
        d: Observed TOD.
        noise_cov_inv: Inverse noise covariance operator.
        solver: Linear solver for the map-making step.

    Returns:
        Scalar negative profile log-likelihood (suitable for minimisation).
    """
    x = (pointing_op.T @ noise_cov_inv)(d)
    ones_tod = ones_like(d)
    h = jax.lax.stop_gradient(pointing_op.T(ones_tod).i)
    h_safe = jnp.where(h > 0, h, 1.0)
    precond = DiagonalOperator(1.0 / h_safe, in_structure=x.structure)
    PtNiP = pointing_op.T @ noise_cov_inv @ pointing_op
    return -dot(x, PtNiP.I(solver=solver, preconditioner=precond)(x))
