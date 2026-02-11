import functools
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntFlag, auto
from typing import Any, ClassVar, TypeVar, overload

if sys.version_info >= (3, 11):
    from typing import dataclass_transform
else:
    from typing_extensions import dataclass_transform

import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jax._src.typing import DType
from jaxtyping import Inexact, PyTree, Scalar, ScalarLike

from furax._config import Config, ConfigState
from furax.tree import zeros_like

from .utils import register_dataclass_with_keys


class OperatorTag(IntFlag):
    """Flags representing properties of linear operators."""

    NONE = 0
    SQUARE = auto()
    SYMMETRIC = auto()
    ORTHOGONAL = auto()
    DIAGONAL = auto()
    TRIDIAGONAL = auto()
    LOWER_TRIANGULAR = auto()
    UPPER_TRIANGULAR = auto()
    POSITIVE_SEMIDEFINITE = auto()
    NEGATIVE_SEMIDEFINITE = auto()


@register_dataclass_with_keys
@dataclass(frozen=True)
@dataclass_transform(frozen_default=True, field_specifiers=(field,))
class AbstractLinearOperator(ABC):
    """Base class for linear operators."""

    # Class-level tags (set by decorators)
    class_tags: ClassVar[OperatorTag] = OperatorTag.NONE

    in_structure: PyTree[jax.ShapeDtypeStruct] = field(
        kw_only=True, metadata={'static': True}, default=None
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        dataclass(frozen=True)(cls)
        register_dataclass_with_keys(cls)

    def __post_init__(self) -> None:
        if self.in_structure is None:
            raise ValueError('The input structure of the operator is not defined.')

    @property
    def tags(self) -> OperatorTag:
        """Get the tags for this operator instance."""
        return self.class_tags

    # Operator properties with default False values
    @property
    def is_square(self) -> bool:
        return bool(self.tags & OperatorTag.SQUARE)

    @property
    def is_symmetric(self) -> bool:
        return bool(self.tags & OperatorTag.SYMMETRIC)

    @property
    def is_orthogonal(self) -> bool:
        return bool(self.tags & OperatorTag.ORTHOGONAL)

    @property
    def is_diagonal(self) -> bool:
        return bool(self.tags & OperatorTag.DIAGONAL)

    @property
    def is_tridiagonal(self) -> bool:
        return bool(self.tags & OperatorTag.TRIDIAGONAL)

    @property
    def is_lower_triangular(self) -> bool:
        return bool(self.tags & OperatorTag.LOWER_TRIANGULAR)

    @property
    def is_upper_triangular(self) -> bool:
        return bool(self.tags & OperatorTag.UPPER_TRIANGULAR)

    @property
    def is_positive_semidefinite(self) -> bool:
        return bool(self.tags & OperatorTag.POSITIVE_SEMIDEFINITE)

    @property
    def is_negative_semidefinite(self) -> bool:
        return bool(self.tags & OperatorTag.NEGATIVE_SEMIDEFINITE)

    @overload
    def __call__(
        self, *, solver: lx.AbstractLinearSolver, **keywords: Any
    ) -> 'AbstractLinearOperator': ...

    @overload
    def __call__(self, x: PyTree[jax.ShapeDtypeStruct]) -> PyTree[jax.ShapeDtypeStruct]: ...

    def __call__(
        self, x: PyTree[jax.ShapeDtypeStruct] | None = None, **keywords: Any
    ) -> 'AbstractLinearOperator | PyTree[jax.ShapeDtypeStruct]':
        if keywords:
            raise TypeError('No keywords is allowed in AbstractLinearOperator __call__ method')
        if isinstance(x, AbstractLinearOperator):
            raise ValueError("Use '@' to compose operators")
        return self.mv(x)

    def __matmul__(self, other: Any) -> 'AbstractLinearOperator':
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        if self.in_structure != other.out_structure:
            raise ValueError('Incompatible linear operator structures')
        if isinstance(other, CompositionOperator):
            return NotImplemented
        if isinstance(other, AbstractLazyInverseOperator):
            if other.operator is self:
                return IdentityOperator(in_structure=self.in_structure)

        return CompositionOperator([self, other])

    def __add__(self, other: Any) -> 'AbstractLinearOperator':
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        if self.in_structure != other.in_structure:
            raise ValueError('Incompatible linear operator input structures')
        if self.out_structure != other.out_structure:
            raise ValueError('Incompatible linear operator output structures')
        if isinstance(other, AdditionOperator):
            return NotImplemented

        return AdditionOperator([self, other])

    def __sub__(self, other: Any) -> 'AbstractLinearOperator':
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        if self.in_structure != other.in_structure:
            raise ValueError('Incompatible linear operator input structures')
        if self.out_structure != other.out_structure:
            raise ValueError('Incompatible linear operator output structures')

        result: AbstractLinearOperator = self + (-other)
        return result

    def __mul__(self, other: ScalarLike) -> 'AbstractLinearOperator':
        result = other * self
        assert isinstance(result, AbstractLinearOperator)  # mypy
        return result

    def __rmul__(self, other: ScalarLike) -> 'AbstractLinearOperator':
        other = jnp.asarray(other)
        if other.shape != ():
            raise ValueError('Can only multiply AbstractLinearOperators by scalars.')
        return HomothetyOperator(other, in_structure=self.out_structure) @ self

    def __truediv__(self, other: ScalarLike) -> 'AbstractLinearOperator':
        other = jnp.asarray(other)
        if other.shape != ():
            raise ValueError('Can only divide AbstractLinearOperators by scalars.')
        return HomothetyOperator(1 / other, in_structure=self.out_structure) @ self

    def __pos__(self) -> 'AbstractLinearOperator':
        return self

    def __neg__(self) -> 'AbstractLinearOperator':
        return (-1) * self

    @abstractmethod
    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]: ...

    def reduce(self) -> 'AbstractLinearOperator':
        """Returns a linear operator with a reduced structure."""
        return self

    def as_matrix(self) -> Inexact[Array, 'a b']:
        """Returns the operator as a dense matrix.

        Input and output PyTrees are flattened and concatenated.
        """
        in_pytree = zeros_like(self.in_structure)
        in_leaves_ref, in_treedef = jax.tree.flatten(in_pytree)

        matrix = jnp.empty((self.out_size, self.in_size), dtype=self.out_promoted_dtype)
        jcounter = 0

        for ileaf, leaf in enumerate(in_leaves_ref):

            def body(index, carry):  # type: ignore[no-untyped-def]
                matrix, jcounter = carry
                zeros = in_leaves_ref.copy()
                zeros[ileaf] = leaf.ravel().at[index].set(1).reshape(leaf.shape)
                in_pytree = jax.tree.unflatten(in_treedef, zeros)
                out_pytree = self.mv(in_pytree)
                out_leaves = [leaf.ravel() for leaf in jax.tree.leaves(out_pytree)]
                matrix = matrix.at[:, jcounter].set(jnp.concatenate(out_leaves))
                jcounter += 1
                return matrix, jcounter

            matrix, jcounter = jax.lax.fori_loop(0, leaf.size, body, (matrix, jcounter))

        return matrix

    def transpose(self) -> 'AbstractLinearOperator':
        return TransposeOperator(self)

    @property
    def T(self) -> 'AbstractLinearOperator':
        return self.transpose()

    def inverse(self) -> 'AbstractLinearOperator':
        return InverseOperator(self)

    @property
    def I(self) -> 'AbstractLinearOperator':  # noqa: E743
        return self.inverse()

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.eval_shape(self.mv, self.in_structure)

    @property
    def in_size(self) -> int:
        """The number of elements in the input PyTree."""
        return sum(_.size for _ in jax.tree.leaves(self.in_structure))

    @property
    def out_size(self) -> int:
        """The number of elements in the output PyTree."""
        return sum(_.size for _ in jax.tree.leaves(self.out_structure))

    @property
    def in_promoted_dtype(self) -> DType[Any]:
        """Returns the promoted data type of the operator's input leaves."""
        leaves = jax.tree.leaves(self.in_structure)
        return jnp.result_type(*leaves)

    @property
    def out_promoted_dtype(self) -> DType[Any]:
        """Returns the promoted data type of the operator's output leaves."""
        leaves = jax.tree.leaves(self.out_structure)
        return jnp.result_type(*leaves)


T = TypeVar('T', bound=AbstractLinearOperator)


def square(cls: type[T]) -> type[T]:
    """Mark an operator as square."""
    cls.class_tags |= OperatorTag.SQUARE
    cls.out_structure = property(lambda self: self.in_structure)  # type: ignore[assignment,method-assign]
    return cls


def symmetric(cls: type[T]) -> type[T]:
    """Mark an operator as symmetric (implies square)."""
    square(cls)
    cls.class_tags |= OperatorTag.SYMMETRIC
    cls.transpose = lambda self: self  # type: ignore[method-assign]
    return cls


def orthogonal(cls: type[T]) -> type[T]:
    """Mark an operator as orthogonal (implies square)."""
    square(cls)
    cls.class_tags |= OperatorTag.ORTHOGONAL
    cls.inverse = cls.transpose  # type: ignore[method-assign]
    return cls


def diagonal(cls: type[T]) -> type[T]:
    """Mark an operator as diagonal (implies symmetric, which implies square)."""
    symmetric(cls)
    cls.class_tags |= OperatorTag.DIAGONAL
    return cls


def tridiagonal(cls: type[T]) -> type[T]:
    """Mark an operator as tridiagonal (implies square)."""
    square(cls)
    cls.class_tags |= OperatorTag.TRIDIAGONAL
    return cls


def lower_triangular(cls: type[T]) -> type[T]:
    """Mark an operator as lower triangular (implies square)."""
    square(cls)
    cls.class_tags |= OperatorTag.LOWER_TRIANGULAR
    return cls


def upper_triangular(cls: type[T]) -> type[T]:
    """Mark an operator as upper triangular (implies square)."""
    square(cls)
    cls.class_tags |= OperatorTag.UPPER_TRIANGULAR
    return cls


def positive_semidefinite(cls: type[T]) -> type[T]:
    """Mark an operator as positive semi-definite (implies square)."""
    square(cls)
    cls.class_tags |= OperatorTag.POSITIVE_SEMIDEFINITE
    return cls


def negative_semidefinite(cls: type[T]) -> type[T]:
    """Mark an operator as negative semi-definite (implies square)."""
    square(cls)
    cls.class_tags |= OperatorTag.NEGATIVE_SEMIDEFINITE
    return cls


class AdditionOperator(AbstractLinearOperator):
    """An operator that adds two operators, as in C = A + B."""

    operands: PyTree[AbstractLinearOperator]

    def __init__(self, operands: PyTree[AbstractLinearOperator]) -> None:
        object.__setattr__(self, 'operands', operands)
        super().__init__(in_structure=self.operand_leaves[0].in_structure)

    # Tag propagation properties
    @property
    def is_square(self) -> bool:
        return super().is_square or self.operand_leaves[0].is_square

    @property
    def is_symmetric(self) -> bool:
        return super().is_symmetric or all(op.is_symmetric for op in self.operand_leaves)

    @property
    def is_diagonal(self) -> bool:
        return super().is_diagonal or all(op.is_diagonal for op in self.operand_leaves)

    @property
    def is_positive_semidefinite(self) -> bool:
        return super().is_positive_semidefinite or all(
            op.is_positive_semidefinite for op in self.operand_leaves
        )

    @property
    def is_negative_semidefinite(self) -> bool:
        return super().is_negative_semidefinite or all(
            op.is_negative_semidefinite for op in self.operand_leaves
        )

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        operands = self.operand_leaves
        y = operands[0](x)

        for operand in operands[1:]:
            y = jax.tree.map(jnp.add, y, operand(x))

        return y

    def transpose(self) -> AbstractLinearOperator:
        return AdditionOperator(self._tree_map(lambda operand: operand.T))

    def __add__(self, other: AbstractLinearOperator) -> 'AdditionOperator':
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        if self.in_structure != other.in_structure:
            raise ValueError('Incompatible linear operator input structures')
        if self.out_structure != other.out_structure:
            raise ValueError('Incompatible linear operator output structures')
        if isinstance(other, AdditionOperator):
            operands = other.operand_leaves
        else:
            operands = [other]
        return AdditionOperator(self.operand_leaves + operands)

    def __radd__(self, other: AbstractLinearOperator) -> 'AdditionOperator':
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        if self.in_structure != other.in_structure:
            raise ValueError('Incompatible linear operator input structures')
        if self.out_structure != other.out_structure:
            raise ValueError('Incompatible linear operator output structures')
        return AdditionOperator([other] + self.operand_leaves)

    def __neg__(self) -> 'AdditionOperator':
        return AdditionOperator(self._tree_map(lambda operand: (-1) * operand))

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operand_leaves[0].out_structure

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return functools.reduce(jnp.add, (operand.as_matrix() for operand in self.operand_leaves))

    def reduce(self) -> AbstractLinearOperator:
        operands = self._tree_map(lambda operand: operand.reduce())
        operand_leaves = jax.tree.leaves(
            operands, is_leaf=lambda leaf: isinstance(leaf, AbstractLinearOperator)
        )
        if len(operand_leaves) == 1:
            leaf: AbstractLinearOperator = operand_leaves[0]
            return leaf
        return AdditionOperator(operands)

    @property
    def operand_leaves(self) -> list[AbstractLinearOperator]:
        """Returns the flat list of operators."""
        return jax.tree.leaves(
            self.operands, is_leaf=lambda x: isinstance(x, AbstractLinearOperator)
        )

    def _tree_map(self, f: Callable[..., Any], *args: Any) -> Any:
        return jax.tree.map(
            f,
            self.operands,
            *args,
            is_leaf=lambda x: isinstance(x, AbstractLinearOperator),
        )


class CompositionOperator(AbstractLinearOperator):
    """An operator that composes two operators, as in C = B ∘ A."""

    operands: list[AbstractLinearOperator]

    def __init__(self, operands: list[AbstractLinearOperator]) -> None:
        object.__setattr__(self, 'operands', operands)
        super().__init__(in_structure=operands[-1].in_structure)

    # Tag propagation properties
    @property
    def is_square(self) -> bool:
        result: bool = super().is_square or (
            self.operands[0].out_structure == self.operands[-1].in_structure
        )
        return result

    @property
    def is_diagonal(self) -> bool:
        return super().is_diagonal or all(op.is_diagonal for op in self.operands)

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        for operand in reversed(self.operands):
            x = operand.mv(x)
        return x

    def transpose(self) -> AbstractLinearOperator:
        return CompositionOperator([_.T for _ in reversed(self.operands)])

    def __matmul__(self, other: AbstractLinearOperator) -> 'CompositionOperator':
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        if self.in_structure != other.out_structure:
            raise ValueError('Incompatible linear operator structures:')
        if isinstance(other, CompositionOperator):
            operands = other.operands
        else:
            operands = [other]
        return CompositionOperator(self.operands + operands)

    def __rmatmul__(self, other: AbstractLinearOperator) -> 'CompositionOperator':
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        if self.out_structure != other.in_structure:
            raise ValueError('Incompatible linear operator structures')
        return CompositionOperator([other] + self.operands)

    def reduce(self) -> AbstractLinearOperator:
        """Returns a linear operator with a reduced structure."""
        from .rules import AlgebraicReductionRule

        operands = AlgebraicReductionRule().apply([operand.reduce() for operand in self.operands])
        if len(operands) == 0:
            return IdentityOperator(in_structure=self.in_structure)
        if len(operands) == 1:
            return operands[0]
        return CompositionOperator(operands)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operands[0].out_structure


class _AbstractLazyDualOperator(AbstractLinearOperator):
    operator: AbstractLinearOperator

    def __post_init__(self) -> None:
        # Here we prefer to use __post_init__ over the __init__ constructor: when __init__ constructors are not
        # present, dataclasses write their own based on the specification of the fields and the resulting constructor
        # does not call the parent constructor.
        # Because of that, once an __init__ method is written in an AbstractLinearOperator, all subclasses must
        # also write their own to explicitly call the parent __init__ constructor.
        # By using the __post_init__ mechanism, subclasses can still define their new fields without having to write
        # an __init__ or __post_init__ method.
        object.__setattr__(self, 'in_structure', self.operator.out_structure)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure


class TransposeOperator(_AbstractLazyDualOperator):
    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        transpose = jax.linear_transpose(self.operator.mv, self.operator.in_structure)
        return transpose(x)[0]

    def transpose(self) -> AbstractLinearOperator:
        return self.operator


class AbstractLazyInverseOperator(_AbstractLazyDualOperator):
    def __call__(
        self, x: PyTree[jax.ShapeDtypeStruct] | None = None, /, **keywords: Any
    ) -> AbstractLinearOperator | PyTree[jax.ShapeDtypeStruct]:
        if x is not None:
            if keywords:
                raise ValueError(
                    'The application of a vector to inverse operator cannot be parametrized. '
                    'For example, instead of A.I(x, throw=True), use A.I(throw=True)(x).'
                )
            return self.mv(x)
        return self

    def __matmul__(self, other: Any) -> AbstractLinearOperator:
        if self.operator is other:
            return IdentityOperator(in_structure=self.in_structure)
        return super().__matmul__(other)

    def inverse(self) -> AbstractLinearOperator:
        return self.operator

    def as_matrix(self) -> Inexact[Array, 'a b']:
        matrix: Array = jnp.linalg.inv(self.operator.as_matrix())
        return matrix


MISSING = object()


class InverseOperator(AbstractLazyInverseOperator):
    config: ConfigState = field(
        kw_only=True, metadata={'static': True}, default_factory=lambda: Config.instance()
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.operator.in_structure != self.operator.out_structure:
            raise ValueError('Only square operators can be inverted.')
        object.__setattr__(self, 'operator', self.operator.reduce())

    def __call__(
        self,
        x: PyTree[jax.ShapeDtypeStruct] | None = None,
        /,
        *,
        solver: lx.AbstractLinearSolver | None = None,
        throw: bool | None = None,
        callback: Callable[[lx.Solution], None] | object = MISSING,
        **options: Any,
    ) -> AbstractLinearOperator | PyTree[jax.ShapeDtypeStruct]:
        config_options = {}
        if solver is not None:
            config_options['solver'] = solver
        if throw is not None:
            config_options['solver_throw'] = throw
        if callback is not MISSING:
            config_options['solver_callback'] = callback
        if options:
            config_options['solver_options'] = options
        if x is None and config_options:
            with Config(**config_options):
                return InverseOperator(self.operator)
        return super().__call__(x, **config_options)

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        from furax.interfaces.lineax import as_lineax_operator

        solver = self.config.solver
        throw = self.config.solver_throw
        options = self.config.solver_options.copy()
        A = as_lineax_operator(self.operator, OperatorTag.POSITIVE_SEMIDEFINITE)
        if preconditioner := options.get('preconditioner'):
            if not isinstance(preconditioner, AbstractLinearOperator):
                raise TypeError('The preconditioner must be an instance of AbstractLinearOperator.')
            options['preconditioner'] = as_lineax_operator(
                preconditioner, OperatorTag.POSITIVE_SEMIDEFINITE
            )
        solution = lx.linear_solve(A, x, solver=solver, throw=throw, options=options)
        jax.debug.callback(self.config.solver_callback, solution)
        return solution.value


@orthogonal
class AbstractLazyInverseOrthogonalOperator(TransposeOperator, AbstractLazyInverseOperator):
    pass


@orthogonal
@diagonal
@positive_semidefinite
class IdentityOperator(AbstractLinearOperator):
    def __matmul__(self, other: Any) -> AbstractLinearOperator:
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        return other

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return x

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return jnp.identity(self.in_size, dtype=self.in_promoted_dtype)


@diagonal
class HomothetyOperator(AbstractLinearOperator):
    value: Scalar | int | float

    def __matmul__(self, other: Any) -> AbstractLinearOperator:
        if isinstance(other, HomothetyOperator):
            return HomothetyOperator(self.value * other.value, in_structure=self.in_structure)
        return super().__matmul__(other)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return jax.tree.map(lambda leaf: self.value * leaf, x)

    def inverse(self) -> AbstractLinearOperator:
        return HomothetyOperator(1 / self.value, in_structure=self.in_structure)

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return self.value * jnp.identity(self.in_size, dtype=self.out_promoted_dtype)
