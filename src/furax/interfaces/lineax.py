from typing import Any

import equinox as eqx
import jax
import lineax as lx
from jax import Array
from jaxtyping import Inexact, PyTree, Shaped

from furax import AbstractLinearOperator, OperatorTag

__all__ = ['as_lineax_operator']


def as_lineax_operator(
    operator: AbstractLinearOperator, tags: OperatorTag = OperatorTag.NONE
) -> lx.AbstractLinearOperator:
    """Wrap a furax operator for use with lineax solvers.

    This function creates a lineax-compatible wrapper that stores the furax
    operator as a pytree field, ensuring JIT compatibility. Tags from the furax
    operator are automatically copied.

    Args:
        operator: A furax AbstractLinearOperator instance

    Returns:
        A lineax AbstractLinearOperator that wraps the furax operator

    Example:
        >>> import furax as fx
        >>> import lineax as lx
        >>> import jax.numpy as jnp
        >>> op = fx.IdentityOperator(in_structure=fx.tree.as_structure(jnp.ones(10)))
        >>> lx_op = fx.as_lineax_operator(op)
        >>> solution = lx.linear_solve(lx_op, jnp.ones(10))
    """
    return _FuraxLinearOperator(
        operator,
        _get_lineax_tags(operator, tags),
    )


def _get_lineax_tags(operator: AbstractLinearOperator, tags: OperatorTag) -> frozenset[Any]:
    """Extract lineax tags from a furax operator."""
    tags |= operator.tags
    lineax_tags = []
    if OperatorTag.SYMMETRIC & tags:
        lineax_tags.append(lx.symmetric_tag)
    if OperatorTag.DIAGONAL & tags:
        lineax_tags.append(lx.diagonal_tag)
    if OperatorTag.TRIDIAGONAL & tags:
        lineax_tags.append(lx.tridiagonal_tag)
    if OperatorTag.LOWER_TRIANGULAR & tags:
        lineax_tags.append(lx.lower_triangular_tag)
    if OperatorTag.UPPER_TRIANGULAR & tags:
        lineax_tags.append(lx.upper_triangular_tag)
    if OperatorTag.POSITIVE_SEMIDEFINITE & tags:
        lineax_tags.append(lx.positive_semidefinite_tag)
    if OperatorTag.NEGATIVE_SEMIDEFINITE & tags:
        lineax_tags.append(lx.negative_semidefinite_tag)
    return frozenset(lineax_tags)


class _FuraxLinearOperator(lx.AbstractLinearOperator):  # type: ignore[misc]
    """Lineax-compatible wrapper around a furax AbstractLinearOperator.

    Unlike lx.FunctionLinearOperator, this stores the furax operator as a
    dynamic pytree field, ensuring JIT compatibility.
    """

    operator: AbstractLinearOperator
    tags: frozenset[Any] = eqx.field(static=True)

    def mv(self, vector: PyTree[Inexact[Array, ' _b']]) -> PyTree[Inexact[Array, ' _a']]:
        return self.operator.mv(vector)

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return self.operator.as_matrix()

    def transpose(self) -> lx.AbstractLinearOperator:
        return _FuraxLinearOperator(
            self.operator.T,
            lx.transpose_tags(self.tags),
        )

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure


def _materialise(operator: _FuraxLinearOperator) -> lx.AbstractLinearOperator:
    return lx.materialise(_to_function_linear_operator(operator))


def _diagonal(operator: _FuraxLinearOperator) -> Shaped[Array, ' size']:
    return lx.diagonal(_to_function_linear_operator(operator))  # type: ignore[no-any-return]


def _tridiagonal(
    operator: _FuraxLinearOperator,
) -> tuple[Shaped[Array, ' size'], Shaped[Array, ' size-1'], Shaped[Array, ' size-1']]:
    return lx.tridiagonal(_to_function_linear_operator(operator))  # type: ignore[no-any-return]


def _to_function_linear_operator(
    operator: _FuraxLinearOperator,
) -> lx.FunctionLinearOperator:
    return lx.FunctionLinearOperator(
        operator.mv, operator.in_structure(), operator.tags, closure_convert=False
    )


lx.materialise.register(_FuraxLinearOperator, _materialise)
lx.diagonal.register(_FuraxLinearOperator, _diagonal)
lx.tridiagonal.register(_FuraxLinearOperator, _tridiagonal)
lx.has_unit_diagonal.register(_FuraxLinearOperator, lambda operator: False)
lx.linearise.register(_FuraxLinearOperator, lambda operator: operator)
lx.conj.register(_FuraxLinearOperator, lambda operator: operator)

# Register singledispatch implementations for lineax query functions.
for _check, _tag in (
    (lx.is_symmetric, lx.symmetric_tag),
    (lx.is_diagonal, lx.diagonal_tag),
    (lx.is_tridiagonal, lx.tridiagonal_tag),
    (lx.is_lower_triangular, lx.lower_triangular_tag),
    (lx.is_upper_triangular, lx.upper_triangular_tag),
    (lx.is_positive_semidefinite, lx.positive_semidefinite_tag),
    (lx.is_negative_semidefinite, lx.negative_semidefinite_tag),
):
    _check.register(_FuraxLinearOperator, lambda operator, tag=_tag: tag in operator.tags)
