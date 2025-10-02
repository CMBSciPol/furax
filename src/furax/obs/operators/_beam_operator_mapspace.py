from typing import cast

import equinox
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jaxtyping import Array, Inexact, PyTree
from scipy.sparse import load_npz

from furax import AbstractLinearOperator, BlockDiagonalOperator, symmetric
from furax.obs.stokes import StokesIQU


def read_beam_matrix(
    path_to_file: str, in_structure: PyTree[jax.ShapeDtypeStruct]
) -> AbstractLinearOperator:
    """Reads a sparse beam matrix from a .npz file and returns a BeamOperatorMapspace class

    Args:
        path_to_file (str): Path to the .npz file containing the sparse beam matrix.
        in_structure (PyTree[jax.ShapeDtypeStruct]): Input structure of the operator.

    Returns:
        BeamOperatorMapspace: Beam operator in map space.
    """

    sparse_matrix = load_npz(path_to_file)
    indices = jnp.array(sparse_matrix.indices, dtype=jnp.int32)
    data = jnp.array(sparse_matrix.data, dtype=jnp.float32)
    n_neighbours = int(jnp.diff(sparse_matrix.indptr)[0])
    n_rows = sparse_matrix.shape[0]

    all_indices = indices.reshape(n_rows, n_neighbours)
    all_data = data.reshape(n_rows, n_neighbours)
    map_structure = in_structure.shape[-1]

    return BeamOperatorMapspace(map_structure, all_indices, all_data)


@symmetric
class BeamOperatorMapspace(AbstractLinearOperator):
    """BeamOperatorMapspace applies a beam to a map in map space.

    Attributes:
        _in_structure (PyTree[jax.ShapeDtypeStruct]): Input structure of the operator.
        _indices (Array): Indices of the CSR format sparse matrix.
        _data (Array): Data of the CSR format sparse matrix.

    Args:
        in_structure (PyTree[jax.ShapeDtypeStruct]): Input structure of the operator.
        indices (Array): Indices of the CSR format sparse matrix.
        data (Array): Data of the CSR format sparse matrix.
    """

    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    _indices: Array = equinox.field(static=False)
    _data: Array = equinox.field(static=False)

    def __init__(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], indices: Array, data: Array
    ) -> None:
        self._in_structure = in_structure
        self._indices = indices
        self._data = data

    def mv(self, x: Inexact[Array, '...']) -> Inexact[Array, '...']:
        return jnp.sum(self._data * x[self._indices], axis=1)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def inverse(self) -> AbstractLinearOperator:
        return BeamOperatorMapspaceInverse(self)


class BeamOperatorMapspaceInverse(AbstractLinearOperator):
    """Inverse of the BeamOperatorMapspace using conjugate gradient method.

    Attributes:
        _beam_operator (BeamOperatorMapspace): The beam operator to be inverted.
        _in_structure (PyTree[jax.ShapeDtypeStruct]): Input structure of the operator.

    Args:
        beam_operator (BeamOperatorMapspace): The beam operator to be inverted.
    """

    _beam_operator: BeamOperatorMapspace
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(self, beam_operator: BeamOperatorMapspace) -> None:
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

    Args:
        axis (int): The axis along which the leaves were originally stacked.
        in_structure (PyTree[jax.ShapeDtypeStruct]): The in_structure of the pytree to be unstacked.
    """

    _axis: int
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        axis: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
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

    Args:
        axis (int): The axis along which the leaves will be stacked.
        in_structure (PyTree[jax.ShapeDtypeStruct]): The in_structure of the pytree to be stacked.
    """

    _axis: int
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        axis: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
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

    Args:
        beam_operators (List[StokesIQU]): List of StokesIQU beam operators to be applied 
            to each frequency.
        in_structure (PyTree[jax.ShapeDtypeStruct]): The in_structure of the input pytree.

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
        _stacked_beam_operator (StackedBeamOperator): The stacked beam operator to be inverted.
        _in_structure (PyTree[jax.ShapeDtypeStruct]): Input structure of the operator.

    Args:
        stacked_beam_operator (StackedBeamOperator): The stacked beam operator to be inverted.
    """

    _stacked_beam_operator: StackedBeamOperator
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(self, stacked_beam_operator: StackedBeamOperator) -> None:
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
