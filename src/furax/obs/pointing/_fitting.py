import dataclasses
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float

from ..coords import tangent_basis
from .modeling import AbstractPointingModel, BasicPointingModel, Observations, _detector_direction

__all__ = ['FitResult', 'fit', 'residuals']

ARCSEC = jnp.pi / (180 * 3600)

type Solver = optx.AbstractLeastSquaresSolver[Any, Any, Any, Any]


class FitResult(eqx.Module):
    """The result of a pointing model fit.

    Attributes:
        model: Best-fit model, including the fixed parameters.
        free: Names of the fitted parameters, the order of `covariance`.
        residuals: Cross-elevation and elevation residuals, shape (..., 2), in arcseconds.
        rms: Root mean square of the residual offsets on the sky, in arcseconds.
        chi2: Sum of the squared weighted residuals.
        dof: Number of degrees of freedom: twice the number of observations minus the number of
            fitted parameters.
        covariance: Covariance matrix of the fitted parameters, in radians squared.
        result: The optimistix status of the solver, e.g. `optimistix.RESULTS.successful`.
        stats: The optimistix solver statistics, e.g. the number of steps.
    """

    model: AbstractPointingModel
    free: tuple[str, ...] = eqx.field(static=True)
    residuals: Float[Array, '... 2']
    rms: Float[Array, '']
    chi2: Float[Array, '']
    dof: int = eqx.field(static=True)
    covariance: Float[Array, 'n_free n_free']
    result: optx.RESULTS
    stats: dict[str, Any]

    @property
    def stderr(self) -> dict[str, Float[Array, '']]:
        """The standard errors of the fitted parameters, in radians."""
        return dict(zip(self.free, jnp.sqrt(jnp.diag(self.covariance))))

    @property
    def correlation(self) -> Float[Array, 'n_free n_free']:
        """The correlation matrix of the fitted parameters."""
        std = jnp.sqrt(jnp.diag(self.covariance))
        return self.covariance / jnp.outer(std, std)


def residuals(model: AbstractPointingModel, obs: Observations) -> Float[Array, '... 2']:
    """The offsets of the predicted from the observed source positions, in radians.

    Returns:
        The cross-elevation and elevation offsets, measured in the plane tangent to the sky at
        the observed position.
    """
    predicted = _detector_direction(model, obs.az, obs.el, obs.roll, obs.q_det)
    e_xel, e_el = tangent_basis(obs.az_obs, obs.el_obs)
    return jnp.stack(
        [jnp.sum(predicted * e_xel, axis=-1), jnp.sum(predicted * e_el, axis=-1)], axis=-1
    )


def fit(
    obs: Observations,
    init: AbstractPointingModel | None = None,
    fixed: Sequence[str] = (),
    solver: Solver | None = None,
    max_steps: int = 256,
    absolute_sigma: bool = False,
) -> FitResult:
    """Fit the pointing model parameters by nonlinear least squares.

    The residuals are those of [`residuals`][], divided by `obs.sigma`. The covariance of the
    parameters is the inverse of $J^T J$, $J$ being the Jacobian of the weighted residuals at the
    solution.

    Args:
        obs: The observations.
        init: Initial model, whose type sets the parameters (a zero `BasicPointingModel` by default).
            Fixed parameters keep their initial values.
        fixed: Names of the parameters that are not fitted, e.g. those that the observations
            cannot separate from others.
        solver: An optimistix least squares solver. By default, Levenberg-Marquardt with a
            relative tolerance of the square root of the machine epsilon, and an absolute
            tolerance of a milliarcsecond in double precision. In single precision, the
            parameters cannot be determined better than about an arcsecond, so enabling
            `jax_enable_x64` is recommended.
        max_steps: Maximum number of solver steps.
        absolute_sigma: If True, `obs.sigma` are absolute uncertainties and the covariance is not
            rescaled by the reduced chi-square.

    Returns:
        The fit result. The fit does not raise if the solver fails: check `result`.
    """
    if init is None:
        init = BasicPointingModel()
    names = init.names()
    unknown = set(fixed) - set(names)
    if unknown:
        raise ValueError(f'Unknown parameters: {sorted(unknown)}')
    free = tuple(name for name in names if name not in fixed)
    if solver is None:
        # the tolerances are limited by the floating point precision of the unit vectors
        eps = float(jnp.finfo(init.to_vector().dtype).eps)
        solver = optx.LevenbergMarquardt(rtol=eps**0.5, atol=max(1e-3 * ARCSEC, 100 * eps))
    return _fit(obs, init, free, solver, max_steps, absolute_sigma)


# jitted at module level, so that repeated fits with the same configuration do not retrace
@eqx.filter_jit
def _fit(
    obs: Observations,
    init: AbstractPointingModel,
    free: tuple[str, ...],
    solver: Solver,
    max_steps: int,
    absolute_sigma: bool,
) -> FitResult:
    sigma = 1.0 if obs.sigma is None else jnp.asarray(obs.sigma)[..., None]

    def to_model(y: Array) -> AbstractPointingModel:
        return dataclasses.replace(init, **dict(zip(free, y)))

    def weighted_residuals(y: Array, obs: Observations) -> Array:
        return (residuals(to_model(y), obs) / sigma).ravel()

    y0 = jnp.stack([getattr(init, name) for name in free])
    solution = optx.least_squares(
        weighted_residuals, solver, y0, args=obs, max_steps=max_steps, throw=False
    )
    y = solution.value

    weighted = weighted_residuals(y, obs)
    jacobian = jax.jacfwd(weighted_residuals)(y, obs)
    chi2 = jnp.sum(weighted**2)
    dof = weighted.size - len(free)
    covariance = jnp.linalg.inv(jacobian.T @ jacobian)
    if not absolute_sigma:
        covariance = covariance * chi2 / dof

    offsets = residuals(to_model(y), obs)
    # the full optimistix Solution holds a jaxpr, which cannot be returned from a jitted function
    return FitResult(
        model=to_model(y),
        free=free,
        residuals=offsets / ARCSEC,
        rms=jnp.sqrt(jnp.mean(jnp.sum(offsets**2, axis=-1))) / ARCSEC,
        chi2=chi2,
        dof=dof,
        covariance=covariance,
        result=solution.result,
        stats=solution.stats,
    )
