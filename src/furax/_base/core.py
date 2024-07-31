from typing import Any, TypeVar

import equinox
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jaxtyping import Float, Inexact, PyTree, ScalarLike

from furax._base.config import Config, ConfigState


class AbstractLinearOperator(lx.AbstractLinearOperator):

    def __init_subclass__(cls, **keywords) -> None:
        _monkey_patch_operator(cls)

    def __call__(self, x: PyTree[jax.ShapeDtypeStruct]) -> PyTree[jax.ShapeDtypeStruct]:
        if isinstance(x, lx.AbstractLinearOperator):
            raise ValueError("Use '@' to compose operators")
        return self.mv(x)

    def __matmul__(self, other: Any) -> 'AbstractLinearOperator':
        if not isinstance(other, lx.AbstractLinearOperator):
            return NotImplemented
        if isinstance(other, CompositionOperator):
            return NotImplemented
        if isinstance(other, AbstractLazyInverseOperator):
            if other.operator is self:
                return IdentityOperator(self.in_structure())
        if self.in_structure() != other.out_structure():
            raise ValueError('Incompatible linear operator structures')

        return CompositionOperator([self, other])

    def __mul__(self, other: ScalarLike) -> 'AbstractLinearOperator':
        return other * self

    def __rmul__(self, other: ScalarLike) -> 'AbstractLinearOperator':
        other = jnp.asarray(other)
        if other.shape != ():
            raise ValueError('Can only multiply AbstractLinearOperators by scalars.')
        return HomothetyOperator(other, self.out_structure()) @ self

    def __truediv__(self, other: ScalarLike) -> 'AbstractLinearOperator':
        other = jnp.asarray(other)
        if other.shape != ():
            raise ValueError('Can only divide AbstractLinearOperators by scalars.')
        return HomothetyOperator(1 / other, self.out_structure()) @ self

    def __pos__(self) -> 'AbstractLinearOperator':
        return self

    def __neg__(self) -> 'AbstractLinearOperator':
        return (-1) * self

    def reduce(self) -> 'AbstractLinearOperator':
        """Returns a linear operator with a reduced structure."""
        return self

    def as_matrix(self) -> Inexact[Array, 'a b']:
        """Returns the operator as a dense matrix.

        Input and output PyTrees are flattened and concatenated.
        """
        in_struct = self.in_structure()
        in_pytree = jax.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), in_struct)
        in_leaves_ref, in_treedef = jax.tree.flatten(in_pytree)

        matrix = jnp.empty(
            (self.out_size(), self.in_size()), dtype=in_leaves_ref[0].dtype
        )  # better dtype ?
        jcounter = 0

        for ileaf, leaf in enumerate(in_leaves_ref):

            def body(index, carry):
                matrix, jcounter = carry
                zeros = in_leaves_ref.copy()
                zeros[ileaf] = leaf.ravel().at[index].set(1).reshape(leaf.shape)
                in_pytree = jax.tree.unflatten(in_treedef, zeros)
                out_pytree = self.mv(in_pytree)
                out_leaves = [l.ravel() for l in jax.tree.leaves(out_pytree)]
                matrix = matrix.at[:, jcounter].set(jnp.concatenate(out_leaves))
                jcounter += 1
                return matrix, jcounter

            matrix, jcounter = jax.lax.fori_loop(0, leaf.size, body, (matrix, jcounter))

        return matrix

    def transpose(self) -> lx.AbstractLinearOperator:
        raise NotImplementedError

    def inverse(self) -> lx.AbstractLinearOperator:
        return LazyInverseOperator(self)

    @property
    def I(self) -> 'AbstractLinearOperator':
        return self.inverse()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        raise NotImplementedError

    def in_size(self) -> int:
        """The number of elements in the input PyTree."""
        return sum(_.size for _ in jax.tree.leaves(self.in_structure()))

    def out_size(self) -> int:
        """The number of elements in the output PyTree."""
        return sum(_.size for _ in jax.tree.leaves(self.out_structure()))


def _monkey_patch_operator(cls: type[lx.AbstractLinearOperator]) -> None:
    """Default tags for the operators"""
    for tag in [
        lx.is_diagonal,
        lx.is_lower_triangular,
        lx.is_upper_triangular,
        lx.is_tridiagonal,
        lx.is_symmetric,
        lx.is_positive_semidefinite,
        lx.is_negative_semidefinite,
    ]:
        if _already_registered(cls, tag):
            continue
        tag.register(cls)(lambda _: False)

    lx.linearise.register(cls)(lambda _: _)
    lx.conj.register(cls)(lambda _: _)


def _monkey_patch_lineax_operator(cls: type[lx.AbstractLinearOperator]) -> None:
    """Patching the lineax operators so that we can call and invert them."""
    _monkey_patch_operator(cls)
    cls.__call__ = AbstractLinearOperator.__call__
    cls.I = AbstractLinearOperator.I
    cls.inverse = AbstractLinearOperator.inverse


def _already_registered(cls: type[lx.AbstractLinearOperator], tag) -> bool:
    return any(
        registered_cls is not object and issubclass(cls, registered_cls)
        for registered_cls in tag.registry
    )


_monkey_patch_lineax_operator(lx.ComposedLinearOperator)


T = TypeVar('T')


def diagonal(cls: type[T]) -> type[T]:
    lx.is_diagonal.register(cls)(lambda _: True)
    symmetric(cls)
    return cls


def lower_triangular(cls: type[T]) -> type[T]:
    lx.is_lower_triangular.register(cls)(lambda _: True)
    square(cls)
    return cls


def upper_triangular(cls: type[T]) -> type[T]:
    lx.is_upper_triangular.register(cls)(lambda _: True)
    square(cls)
    return cls


def symmetric(cls: type[T]) -> type[T]:
    lx.is_symmetric.register(cls)(lambda _: True)
    square(cls)
    cls.transpose = lambda self: self
    return cls


def positive_semidefinite(cls: type[T]) -> type[T]:
    lx.is_positive_semidefinite.register(cls)(lambda _: True)
    square(cls)
    return cls


def negative_semidefinite(cls: type[T]) -> type[T]:
    lx.is_negative_semidefinite.register(cls)(lambda _: True)
    square(cls)
    return cls


# not a lineax tag
def square(cls: type[T]) -> type[T]:
    cls.out_structure = cls.in_structure
    return cls


# not a lineax tag
def orthogonal(cls: type[T]) -> type[T]:
    square(cls)
    cls.inverse = cls.transpose
    return cls


class CompositionOperator(AbstractLinearOperator):
    operands: list[AbstractLinearOperator]

    def __init__(self, operands: list[AbstractLinearOperator]) -> None:
        self.operands = operands

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        reduced_operator = self.reduce()
        if isinstance(reduced_operator, CompositionOperator):
            for operator in reversed(reduced_operator.operands):
                x = operator.mv(x)
            return x
        return reduced_operator.mv(x)

    def transpose(self) -> AbstractLinearOperator:
        return CompositionOperator([_.T for _ in reversed(self.operands)])

    def __matmul__(self, other: AbstractLinearOperator) -> 'CompositionOperator':
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        if self.in_structure() != other.out_structure():
            raise ValueError('Incompatible linear operator structures:')
        if isinstance(other, CompositionOperator):
            operands = other.operands
        else:
            operands = [other]
        return CompositionOperator(self.operands + operands)

    def __rmatmul__(self, other: AbstractLinearOperator) -> 'CompositionOperator':
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        if self.out_structure() != other.in_structure():
            raise ValueError('Incompatible linear operator structures')
        return CompositionOperator([other] + self.operands)

    def reduce(self) -> AbstractLinearOperator:
        """Returns a linear operator with a reduced structure."""
        from .rules import AlgebraicReductionRule

        operands = AlgebraicReductionRule().apply(self.operands)
        if len(operands) == 0:
            return IdentityOperator(self.in_structure())
        if len(operands) == 1:
            return operands[0]
        return CompositionOperator(operands)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operands[-1].in_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operands[0].out_structure()


class _AbstractLazyDualOperator(AbstractLinearOperator):  # type: ignore[misc]
    operator: AbstractLinearOperator

    def __init__(self, operator: AbstractLinearOperator):
        self.operator = operator

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure()


class AbstractLazyTransposeOperator(_AbstractLazyDualOperator):  # type: ignore[misc]

    def transpose(self) -> AbstractLinearOperator:
        return self.operator

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return self.operator.as_matrix().T


class AbstractLazyInverseOperator(_AbstractLazyDualOperator):  # type: ignore[misc]

    def __matmul__(self, other):
        if self.operator is other:
            return IdentityOperator(self.in_structure())
        return super().__matmul__(other)

    def inverse(self) -> AbstractLinearOperator:
        return self.operator

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return jnp.linalg.inv(self.operator.as_matrix())


class LazyInverseOperator(AbstractLazyInverseOperator):
    config: ConfigState = equinox.field(static=True)

    def __init__(self, operator: AbstractLinearOperator):
        if operator.in_structure() != operator.out_structure():
            raise ValueError('Only square operators can be inverted.')
        super().__init__(operator)
        self.config = Config.instance()

    def mv(self, x):
        reduced_operator = self.operator.reduce()
        solver = self.config.solver
        throw = self.config.solver_throw
        options = self.config.solver_options.copy()
        A = lx.TaggedLinearOperator(reduced_operator, lx.positive_semidefinite_tag)
        if preconditioner := options.get('preconditioner'):
            options['preconditioner'] = lx.TaggedLinearOperator(
                preconditioner, lx.positive_semidefinite_tag
            )
        solution = lx.linear_solve(A, x, solver=solver, throw=throw, options=options)
        _ = jax.experimental.io_callback(
            self.config.solver_callback, jax.ShapeDtypeStruct((), jnp.int8), solution
        )
        return solution.value


@orthogonal
class AbstractLazyInverseOrthogonalOperator(AbstractLazyTransposeOperator, AbstractLazyInverseOperator):  # type: ignore[misc]
    pass


@orthogonal
@diagonal
class IdentityOperator(AbstractLinearOperator):  # type: ignore[misc]
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(self, in_structure: PyTree[jax.ShapeDtypeStruct]):
        self._in_structure = in_structure

    def __matmul__(self, other):
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        return other

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return x

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return jnp.identity(self.in_size())


@diagonal
class HomothetyOperator(AbstractLinearOperator):  # type: ignore[misc]
    value: float
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(self, value: ScalarLike, in_structure: PyTree[jax.ShapeDtypeStruct]):
        self.value = value
        self._in_structure = in_structure

    def __matmul__(self, other):
        if isinstance(other, HomothetyOperator):
            return HomothetyOperator(self.value * other.value, self._in_structure)
        return super().__matmul__(other)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return jax.tree_map(lambda leave: self.value * leave, x)

    def inverse(self):
        return HomothetyOperator(1 / self.value, self._in_structure)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return self.value * jnp.identity(self.in_size())


@diagonal
class DiagonalOperator(AbstractLinearOperator):  # type: ignore[misc]
    diagonal: PyTree[Float[Array, '...']] = equinox.field(static=True)

    def __init__(self, diagonal: PyTree[Float[Array, '...']]):
        self.diagonal = diagonal

    def mv(self, sky: PyTree[Float[Array, '...']]) -> PyTree[Float[Array, '...']]:
        return jax.tree_map((lambda a, b: a * b), sky, self.diagonal)

    def inverse(self) -> 'AbstractLinearOperator':
        return DiagonalInverseOperator(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.eval_shape(lambda: self.diagonal)

    def as_matrix(self) -> Inexact[Array, 'a b']:
        leaves = jax.tree.leaves(self.diagonal)
        return jnp.diag(jnp.concatenate(jax.tree_map(lambda _: _.ravel(), leaves)))


class DiagonalInverseOperator(AbstractLazyInverseOperator):
    def mv(self, x: PyTree[Float[Array, '...']]) -> PyTree[Float[Array, '...']]:
        def mul_inv(diagonal_leaf, x_leaf):
            return jnp.where(diagonal_leaf != 0, x_leaf / diagonal_leaf, 0)

        y = jax.tree_map(mul_inv, self.operator.diagonal, x)
        return y

    def as_matrix(self) -> Inexact[Array, 'a b']:
        def inv(diagonal_leaf):
            leaf = diagonal_leaf.ravel()
            return jnp.where(leaf != 0, 1 / leaf, 0)

        leaves = jax.tree.leaves(self.operator.diagonal)
        return jnp.diag(jnp.concatenate(jax.tree_map(inv, leaves)))
