r"""The weighted least-squares fit of templates to a TOD.

Fitting the templates $T$ to a TOD $d$ under a weight $W$ gives the amplitudes
$\hat a = (T^\top W T)^{-1} T^\top W d$. [`TemplateFit`][] solves these normal equations by
conjugate gradients, preconditioned by the inverse of the Gram of the templates
([`template_gram`][furax.mapmaking.templates.gram.template_gram] and
[`factor`][furax.mapmaking.templates.gram.factor]), and applies to as many TODs as needed.
"""

from functools import partial
from typing import Any

import jax
from jaxtyping import Array

from furax import AbstractLinearOperator
from furax.linalg import CGResult, CGSolver
from furax.obs import TOD

from ._operator import TemplateOperator
from .gram import factor, template_gram

__all__ = ['TemplateFit']


class TemplateFit:
    r"""The weighted least-squares fit of templates to TODs.

    The Gram is assembled and factored once, at construction. Calling the fit on a TOD then solves
    its normal equations, so one fit serves any number of TODs of the same format.

    With a `regularization`, the fit is the ridge regression whose penalty is that of
    [`factor`][furax.mapmaking.templates.gram.factor], which also keeps the Gram of amplitudes that
    no weighted sample constrains invertible.

    Args:
        operator: The template operator $T$.
        weight: The weight $W$ on the TOD.
        regularization: The relative ridge of the Gram blocks; 0 for none.
        batch_size: The number of detectors, or probes, the Gram is assembled from at once.
        solver: The conjugate gradient settings, by default those of
            [`CGSolver`][furax.linalg.CGSolver].
        **gram_options: The other options of
            [`template_gram`][furax.mapmaking.templates.gram.template_gram], such as `bandwidth`
            for a weight that couples samples, or `pomme_tau`.

    Examples:
        Two detectors, each with a linear drift and an unknown offset, and a flagged sample:

        >>> import jax.numpy as jnp
        >>> from furax.mapmaking.templates import Template, TemplateOperator
        >>> from furax.mapmaking.templates.bases import TensorBasis
        >>> from furax.obs import TODFormat
        >>> tod_format = TODFormat((2, 4), jnp.float32)
        >>> drift = TensorBasis(jnp.stack([jnp.ones(4), jnp.arange(4.0)]))
        >>> operator = TemplateOperator({'drift': Template(drift, tod_format)})
        >>> weight = tod_format.diagonal(jnp.ones((1, 2, 4)).at[:, :, 1].set(0))
        >>> fit = TemplateFit(operator, weight)
        >>> amplitudes = {'drift': jnp.array([[1.0, 2.0], [3.0, -1.0]])}
        >>> result = fit(operator(amplitudes))
        >>> bool(jnp.allclose(result.solution['drift'], amplitudes['drift'], atol=1e-4))
        True
    """

    def __init__(
        self,
        operator: TemplateOperator,
        weight: AbstractLinearOperator,
        *,
        regularization: float = 0.0,
        batch_size: int = 1,
        solver: CGSolver | None = None,
        **gram_options: Any,
    ) -> None:
        self.operator = operator
        self.solver = solver or CGSolver(rtol=1e-6)
        gram = template_gram(operator, weight, batch_size=batch_size, **gram_options)
        self._preconditioner, ridge = factor(gram, regularization=regularization)
        self._project = operator.T @ weight
        normal = self._project @ operator
        self._normal = normal if ridge is None else (normal + ridge).reduce()

    def __call__(self, tod: TOD) -> CGResult:
        """Fit the templates to `tod`: the amplitudes are the `solution` of the result."""
        return _solve(self.solver, self._normal, self._project(tod), self._preconditioner)

    def model(self, amplitudes: dict[str, Array]) -> TOD:
        """The TOD of the templates with these amplitudes."""
        return self.operator(amplitudes)


@partial(jax.jit, static_argnums=0)
def _solve(
    solver: CGSolver,
    normal: AbstractLinearOperator,
    rhs: Any,
    preconditioner: AbstractLinearOperator,
) -> CGResult:
    return solver(normal, rhs, preconditioner=preconditioner)
