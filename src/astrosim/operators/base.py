from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jaxtyping import Array, Float, Inexact, PyTree


class AbstractLinearOperator(lx.AbstractLinearOperator):
    def __init_subclass__(cls, **keywords) -> None:
        _monkey_patch_operator(cls)

    def __call__(self, x: PyTree[jax.ShapeDtypeStruct]) -> PyTree[jax.ShapeDtypeStruct]:
        # just for the static type checkers, it is overriden by the monkey patch.
        raise NotImplementedError

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

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        raise NotImplementedError

    def in_size(self) -> int:
        """The number of elements in the input PyTree."""
        return sum(_.size for _ in jax.tree.leaves(self.in_structure()))

    def out_size(self) -> int:
        """The number of elements in the output PyTree."""
        return sum(_.size for _ in jax.tree.leaves(self.out_structure()))


def _monkey_patch_operator(cls: type[lx.AbstractLinearOperator]) -> None:
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

    cls.__call__ = _base_class__call__


def _already_registered(cls: type[lx.AbstractLinearOperator], tag) -> bool:
    return any(
        registered_cls is not object and issubclass(cls, registered_cls)
        for registered_cls in tag.registry
    )


def _base_class__call__(self, x: PyTree[jax.ShapeDtypeStruct]) -> PyTree[jax.ShapeDtypeStruct]:
    if isinstance(x, lx.AbstractLinearOperator):
        raise ValueError("Use '@' to compose operators")
    return self.mv(x)


_monkey_patch_operator(lx.ComposedLinearOperator)


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


@diagonal
class IdentityOperator(AbstractLinearOperator):  # type: ignore[misc]
    _in_structure: PyTree[jax.ShapeDtypeStruct]

    def __init__(self, in_structure: PyTree[jax.ShapeDtypeStruct]):
        self._in_structure = in_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return x

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return jnp.identity(self.in_size())

    def __matmul__(self, other):
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        return other

    def __rmatmul__(self, other):
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        return other


@diagonal
class HomothetyOperator(AbstractLinearOperator):  # type: ignore[misc]
    _in_structure: PyTree[jax.ShapeDtypeStruct]
    value: float

    def __init__(self, in_structure: PyTree[jax.ShapeDtypeStruct], value: float):
        self._in_structure = in_structure
        self.value = value

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return jax.tree_map(lambda leave: self.value * leave, x)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return self.value * jnp.identity(self.in_size())

    def __matmul__(self, other):
        if not isinstance(other, AbstractLinearOperator):
            return NotImplemented
        return other


@diagonal
class DiagonalOperator(AbstractLinearOperator):  # type: ignore[misc]
    diagonal: PyTree[Float[Array, '...']] = eqx.field(static=True)

    def __init__(self, diagonal: PyTree[Float[Array, '...']]):
        self.diagonal = diagonal

    def mv(self, sky: PyTree[Float[Array, '...']]) -> PyTree[Float[Array, '...']]:
        return jax.tree_map((lambda a, b: a * b), sky, self.diagonal)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.eval_shape(lambda: self.diagonal)

    def as_matrix(self) -> Inexact[Array, 'a b']:
        leaves = jax.tree.leaves(self.diagonal)
        return jnp.diag(jnp.concatenate(jax.tree_map(lambda _: _.ravel(), leaves)))
