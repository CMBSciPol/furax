from pathlib import Path
from typing import cast

import equinox
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jaxtyping import Array, Inexact, PyTree
from typing_extensions import Self

from furax import (
    AbstractLinearOperator,
    BlockDiagonalOperator,
    square,
)
from furax.obs.stokes import StokesIQU


@square
class MapSpaceBeamOperator(AbstractLinearOperator):
    """MapSpaceBeamOperator applies a beam to a map in map space.

    Attributes:
        _in_structure: Input structure of the operator.
        _indices: Indices of the CSR format sparse matrix.
        _data: Data of the CSR format sparse matrix.
    """

    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    indices: Array
    data: Array
    observed_pixels: Array

    def __init__(
        self,
        indices: Array,
        data: Array,
        observed_pixels: Array,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        """
        Args:
            indices: Indices of the CSR format sparse matrix.
            data: Data of the CSR format sparse matrix.
            in_structure: Input structure of the operator.
        """
        self._in_structure = in_structure
        self.indices = indices
        self.data = data
        self.observed_pixels = observed_pixels

    def mv(self, x: Inexact[Array, '...']) -> Inexact[Array, '...']:
        result = jnp.zeros_like(x)
        observed_values = jnp.sum(self.data * x[self.indices], axis=1)
        return result.at[self.observed_pixels].set(observed_values, unique_indices=True)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def inverse(self) -> AbstractLinearOperator:
        return MapSpaceBeamOperatorInverse(self)

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        sparse_matrix = jnp.load(filename)
        indices = jnp.array(sparse_matrix['indices'], dtype=jnp.int32)
        data = jnp.array(sparse_matrix['data'], dtype=jnp.float32)
        observed_pixels = jnp.array(sparse_matrix['observed_pixels'], dtype=jnp.int32)

        n_neighbours = sparse_matrix['max_nnz']
        n_rows = len(observed_pixels)
        nside = sparse_matrix['nside']
        npix = 12 * nside**2

        all_indices = indices.reshape(n_rows, n_neighbours)
        all_data = data.reshape(n_rows, n_neighbours)
        in_structure = jax.ShapeDtypeStruct((npix,), data.dtype)

        return cls(all_indices, all_data, observed_pixels, in_structure=in_structure)


@square
class MapSpaceBeamOperatorInverse(AbstractLinearOperator):
    """Inverse of the MapSpaceBeamOperator using conjugate gradient method.

    Attributes:
        _beam_operator: The beam operator to be inverted.
        _in_structure: Input structure of the operator.
    """

    _beam_operator: MapSpaceBeamOperator
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(self, beam_operator: MapSpaceBeamOperator) -> None:
        """
        Args:
            beam_operator (MapSpaceBeamOperator): The beam operator to be inverted.
        """
        self._beam_operator = beam_operator
        self._in_structure = beam_operator.in_structure()

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: Inexact[Array, '...']) -> Inexact[Array, '...']:
        def matvec(v: Inexact[Array, '...']) -> Inexact[Array, '...']:
            return self._beam_operator.mv(v)

        x_sol, _ = cg(matvec, x)
        return cast(Inexact[Array, '...'], x_sol)


class StokesToListOperator(AbstractLinearOperator):
    """Operator that converts StokesIQU[Array[n,m]] to [StokesIQU[Array[m]], ...].

    Note:
        Index notation: x[stokes,(freq,pix)] -> x[freq,stokes,(pix)]

    Attributes:
        _axis: The axis along which the leaves were originally stacked.
        _in_structure: The in_structure of the pytree to be unstacked.
    """

    _axis: int
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        axis: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        """
        Args:
            axis: The axis along which the leaves were originally stacked.
            in_structure: The in_structure of the pytree to be unstacked.
        """
        self._axis = axis
        self._in_structure = in_structure

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return [
            StokesIQU(
                i=jnp.take(jnp.atleast_2d(x.i), f, axis=self._axis),
                q=jnp.take(jnp.atleast_2d(x.q), f, axis=self._axis),
                u=jnp.take(jnp.atleast_2d(x.u), f, axis=self._axis),
            )
            for f in range(jnp.atleast_2d(x.i).shape[self._axis])
        ]


class ListToStokesOperator(AbstractLinearOperator):
    """Operator that converts [StokesIQU[Array[m]], ...] to StokesIQU[Array[n,m]].

    Note:
        Index notation: x[freq,stokes,(pix)] -> x[stokes,(freq,pix)]

    Attributes:
        _axis: The axis along which the leaves will be stacked.
        _in_structure: The in_structure of the pytree to be stacked.
    """

    _axis: int
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        axis: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        """
        Args:
            axis: The axis along which the leaves will be stacked.
            in_structure: The in_structure of the pytree to be stacked.
        """
        self._axis = axis
        self._in_structure = in_structure

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return StokesIQU(
            i=jnp.stack([s.i for s in x], axis=self._axis),
            q=jnp.stack([s.q for s in x], axis=self._axis),
            u=jnp.stack([s.u for s in x], axis=self._axis),
        )


@square
class StackedBeamOperator(AbstractLinearOperator):
    """Operator that applies a list of beam operators to a stacked StokesIQU map.

    Note:
        Index notation: x[stokes,(freq,pix)] -> x[stokes,(freq,pix)]

    Attributes:
        _beam_operators: List of StokesIQU beam operators to be applied to each frequency.
        _in_structure: The in_structure of the input pytree.

    Example:
        d = ... some StokeIQU object with shape (3, N_freq, N_pix)

        # load files
        B_f090_i = read_beam_matrix(in_structure=d.structure, path_to_file='beam_sparse_f090_I.npz')
        B_f090_q = read_beam_matrix(in_structure=d.structure, path_to_file='beam_sparse_f090_Q.npz')
        B_f090_u = read_beam_matrix(in_structure=d.structure, path_to_file='beam_sparse_f090_U.npz')

        B_f150_i = read_beam_matrix(in_structure=d.structure, path_to_file='beam_sparse_f150_I.npz')
        B_f150_q = read_beam_matrix(in_structure=d.structure, path_to_file='beam_sparse_f150_Q.npz')
        B_f150_u = read_beam_matrix(in_structure=d.structure, path_to_file='beam_sparse_f150_U.npz')

        beam_operators = [
            StokesIQU(i = B_f090_i, q = B_f090_q, u = B_f090_u),
            StokesIQU(i = B_f150_i, q = B_f150_q, u = B_f150_u),
        ]

        B = StackedBeamOperator(beam_operators=beam_operators, in_structure=d.structure)
        d_beam_applied = B(d)
    """

    _beam_operators: PyTree
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        beam_operators: PyTree,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        """
        Args:
            beam_operators: List of StokesIQU beam operators to be applied to each frequency.
            in_structure: The in_structure of the input pytree.
        """
        self._beam_operators = beam_operators
        self._in_structure = in_structure

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        d_list = StokesToListOperator(axis=0, in_structure=self.in_structure()).mv(x)
        n_freq = len(d_list)
        B = [BlockDiagonalOperator(self._beam_operators[f]) for f in range(n_freq)]
        r_list = [B[f].mv(d_list[f]) for f in range(n_freq)]
        return ListToStokesOperator(axis=0, in_structure=self.in_structure()).mv(r_list)

    def inverse(self) -> AbstractLinearOperator:
        return StackedBeamOperatorInverse(self)


@square
class StackedBeamOperatorInverse(AbstractLinearOperator):
    """Inverse of the StackedBeamOperator using conjugate gradient method for each frequency.

    Attributes:
        _stacked_beam_operator: The stacked beam operator to be inverted.
        _in_structure: Input structure of the operator.
    """

    _stacked_beam_operator: StackedBeamOperator
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(self, stacked_beam_operator: StackedBeamOperator) -> None:
        """
        Args:
            stacked_beam_operator: The stacked beam operator to be inverted.
        """
        self._stacked_beam_operator = stacked_beam_operator
        self._in_structure = stacked_beam_operator.in_structure()

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        d_list = StokesToListOperator(axis=0, in_structure=self.in_structure()).mv(x)
        n_freq = len(d_list)

        inverse_beam_operators = []
        for f in range(n_freq):
            beam_op_freq = self._stacked_beam_operator._beam_operators[f]
            inverse_beam_operators.append(
                StokesIQU(i=beam_op_freq.i.I, q=beam_op_freq.q.I, u=beam_op_freq.u.I)
            )

        B = [BlockDiagonalOperator(inverse_beam_operators[f]) for f in range(n_freq)]
        r_list = [B[f].mv(d_list[f]) for f in range(n_freq)]

        return ListToStokesOperator(axis=0, in_structure=self.in_structure()).mv(r_list)
