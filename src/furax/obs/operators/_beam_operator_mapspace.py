from typing import cast

import equinox
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jaxtyping import Array, Inexact, PyTree

from furax import AbstractLinearOperator, BlockDiagonalOperator, symmetric
from furax.obs.stokes import StokesIQU


def read_beam_matrix(
    path_to_file: str, in_structure: PyTree[jax.ShapeDtypeStruct]
) -> AbstractLinearOperator:
    """Reads a sparse beam matrix from a .npz file and returns a MapSpaceBeamOperator class

    Args:
        path_to_file: Path to the .npz file containing the sparse beam matrix.
        in_structure: Input structure of the operator.

    Returns:
        MapSpaceBeamOperator: Beam operator in map space.
    """

    sparse_matrix = jnp.load(path_to_file)
    indices = jnp.array(sparse_matrix['indices'], dtype=jnp.int32)
    data = jnp.array(sparse_matrix['data'], dtype=jnp.float32)
    n_neighbours = int(jnp.diff(sparse_matrix['indptr'])[0])
    n_rows = sparse_matrix['indptr'].shape[0] - 1

    all_indices = indices.reshape(n_rows, n_neighbours)
    all_data = data.reshape(n_rows, n_neighbours)
    map_structure = in_structure.shape[-1]

    return MapSpaceBeamOperator(map_structure, all_indices, all_data)


# @symmetric
class MapSpaceBeamOperator(AbstractLinearOperator):
    """MapSpaceBeamOperator applies a beam to a map in map space.

    Attributes:
        _in_structure: Input structure of the operator.
        _indices: Indices of the CSR format sparse matrix.
        _data: Data of the CSR format sparse matrix.
    """

    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    _indices: Array = equinox.field(static=False)
    _data: Array = equinox.field(static=False)

    def __init__(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], indices: Array, data: Array
    ) -> None:
        """
        Args:
            in_structure: Input structure of the operator.
            indices: Indices of the CSR format sparse matrix.
            data: Data of the CSR format sparse matrix.
        """
        self._in_structure = in_structure
        self._indices = indices
        self._data = data

    def mv(self, x: Inexact[Array, '...']) -> Inexact[Array, '...']:
        return jnp.sum(self._data * x[self._indices], axis=1)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def inverse(self) -> AbstractLinearOperator:
        return MapSpaceBeamOperatorInverse(self)


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
                i=jnp.take(x.i, f, axis=self._axis),
                q=jnp.take(x.q, f, axis=self._axis),
                u=jnp.take(x.u, f, axis=self._axis),
            )
            for f in range(x.i.shape[self._axis])
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


@symmetric
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
