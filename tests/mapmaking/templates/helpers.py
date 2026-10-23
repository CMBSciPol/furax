import jax.numpy as jnp
from jax import Array

from furax.mapmaking.templates import Template, TemplateOperator
from furax.mapmaking.templates.bases import Basis
from furax.obs import TODFormat

# float64 everywhere: structural/adjoint residuals run ~1e-14, so one tight tolerance fits all.
TOL = 1e-12


def wrap(basis: Basis, n_dets: int) -> TemplateOperator:
    """Wrap a bare basis as a single-template TemplateOperator: amplitudes are {'t': ...},
    output is a bare (n_dets, samp) array."""
    tod_format = TODFormat((n_dets, basis.n_points), basis.dtype)
    return TemplateOperator({'t': Template(basis, tod_format)})


def adjoint_residual(basis: Basis, coeffs: Array, signal: Array) -> float:
    """|<expand(coeffs), signal> - <coeffs, project(signal)>|."""
    lhs = jnp.vdot(basis(coeffs), signal)
    rhs = jnp.vdot(coeffs, basis.T(signal))
    return float(jnp.abs(lhs - rhs))
