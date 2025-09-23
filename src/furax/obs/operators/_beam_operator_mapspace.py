from typing import cast

import equinox
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jaxtyping import Array, Inexact, PyTree
from scipy.sparse import load_npz

from furax import AbstractLinearOperator, BlockDiagonalOperator, symmetric
from furax.obs.stokes import StokesIQU


def ReadBeamMatrix(
    path_to_file: str, in_structure: PyTree[jax.ShapeDtypeStruct]
) -> AbstractLinearOperator:
    """
    Reads a sparse beam matrix from a .npz file and returns a BeamOperatorMapspace class
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
    """
    BeamOperatorMapspace applies a beam to a map in map space.

    Args:
        _in_structure (PyTree[jax.ShapeDtypeStruct]): Input structure of the operator.
        _indices (Array): Indices of the sparse matrix in CSR format.
        _data (Array): Data of the sparse matrix in CSR format.
    Returns:
        Array: The result of applying the beam operator to the input map.
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
    """
    Inverse of the BeamOperatorMapspace using conjugate gradient method.

    Args:
        beam_operator (BeamOperatorMapspace): The beam operator to be inverted.
    Returns:
        Array: The result of applying the inverse beam operator to the input map.
    """

    beam_operator: BeamOperatorMapspace
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(self, beam_operator: BeamOperatorMapspace) -> None:
        self.beam_operator = beam_operator
        self._in_structure = beam_operator.in_structure()

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: Inexact[Array, '...']) -> Inexact[Array, '...']:
        # Solve Ax = b using jax.scipy.sparse.linalg.cg
        A = self.beam_operator

        def matvec(v: Inexact[Array, '...']) -> Inexact[Array, '...']:
            return A.mv(v)

        x_sol, info = cg(matvec, x)
        return cast(Inexact[Array, '...'], x_sol)


class StokesToListOperator(AbstractLinearOperator):
    """
    Operator that converts StokesIQU[Array[n,m]] to [StokesIQU[Array[m]], ...].
    Index notation: x[stokes,(freq,pix)] -> x[freq,stokes,(pix)]

    Attributes:
        axis: The axis along which the leaves were originally stacked.
        in_structure: The in_structure of the pytree to be unstacked.
    Returns:
        List[StokesIQU[Array[m]]]: List of StokesIQU objects for each frequency.
    """

    axis: int
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        axis: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        self.axis = axis
        self._in_structure = in_structure

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return [
            StokesIQU(
                i=jnp.take(x.i, f, axis=self.axis),
                q=jnp.take(x.q, f, axis=self.axis),
                u=jnp.take(x.u, f, axis=self.axis),
            )
            for f in range(x.i.shape[self.axis])
        ]


class ListToStokesOperator(AbstractLinearOperator):
    """
    Operator that converts [StokesIQU[Array[m]], ...] to StokesIQU[Array[n,m]].
    Index notation: x[freq,stokes,(pix)] -> x[stokes,(freq,pix)]

    Attributes:
        axis: The axis along which the leaves will be stacked.
        in_structure: The in_structure of the pytree to be stacked.
    Returns:
        StokesIQU[Array[n,m]]: StokesIQU object with stacked frequencies.
    """

    axis: int
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        axis: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        self.axis = axis
        self._in_structure = in_structure

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return StokesIQU(
            i=jnp.stack([s.i for s in x], axis=self.axis),
            q=jnp.stack([s.q for s in x], axis=self.axis),
            u=jnp.stack([s.u for s in x], axis=self.axis),
        )


@symmetric
class StackedBeamOperator(AbstractLinearOperator):
    """
    Operator that applies a list of beam operators to a stacked StokesIQU map.
    Index notation: x[stokes,(freq,pix)] -> x[stokes,(freq,pix)]

    Attributes:
        beam_operators: List of StokesIQU beam operators to be applied to each frequency.
        in_structure: The in_structure of the input pytree.
    Returns:
        StokesIQU[Array[n,m]]: The result of applying the beam operators to the input map.

    Example:
        d = ... some StokeIQU object with shape (3, N_freq, N_pix)

        # load files
        B_f090_i = ReadBeamMatrix(in_structure=d.structure, path_to_file='beam_sparse_f090_I.npz')
        B_f090_q = ReadBeamMatrix(in_structure=d.structure, path_to_file='beam_sparse_f090_Q.npz')
        B_f090_u = ReadBeamMatrix(in_structure=d.structure, path_to_file='beam_sparse_f090_U.npz')

        B_f150_i = ReadBeamMatrix(in_structure=d.structure, path_to_file='beam_sparse_f150_I.npz')
        B_f150_q = ReadBeamMatrix(in_structure=d.structure, path_to_file='beam_sparse_f150_Q.npz')
        B_f150_u = ReadBeamMatrix(in_structure=d.structure, path_to_file='beam_sparse_f150_U.npz')

        beam_operators = [
            StokesIQU(i = B_f090_i, q = B_f090_q, u = B_f090_u),
            StokesIQU(i = B_f150_i, q = B_f150_q, u = B_f150_u),
        ]

        B = StackedBeamOperator(beam_operators=beam_operators, in_structure=d.structure)
        d_beamed = B(d)
    """

    beam_operators: PyTree
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        beam_operators: PyTree,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        self.beam_operators = beam_operators
        self._in_structure = in_structure

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        d_list = StokesToListOperator(axis=0, in_structure=self.in_structure()).mv(x)
        N_freq = len(d_list)
        B_freq = [BlockDiagonalOperator(self.beam_operators[f]) for f in range(N_freq)]
        r_list = [B_freq[f].mv(d_list[f]) for f in range(N_freq)]
        return ListToStokesOperator(axis=0, in_structure=self.in_structure()).mv(r_list)

    def inverse(self) -> AbstractLinearOperator:
        return StackedBeamOperatorInverse(self)


class StackedBeamOperatorInverse(AbstractLinearOperator):
    """
    Inverse of the StackedBeamOperator using conjugate gradient method for each frequency.
    Args:
        stacked_beam_operator (StackedBeamOperator): The stacked beam operator to be inverted.
    Returns:
        PyTree[Inexact[Array, '...']]: The result of applying the inverse stacked beam operator.
    """

    stacked_beam_operator: StackedBeamOperator
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(self, stacked_beam_operator: StackedBeamOperator) -> None:
        self.stacked_beam_operator = stacked_beam_operator
        self._in_structure = stacked_beam_operator.in_structure()

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        # Split into per-frequency components
        d_list = StokesToListOperator(axis=0, in_structure=self.in_structure()).mv(x)
        N_freq = len(d_list)

        # Create inverse beam operators for each frequency
        inverse_beam_operators = []
        for f in range(N_freq):
            beam_op_freq = self.stacked_beam_operator.beam_operators[f]
            inverse_beam_operators.append(
                StokesIQU(i=beam_op_freq.i.I, q=beam_op_freq.q.I, u=beam_op_freq.u.I)
            )

        # Apply the inverse operators using the same pattern as StackedBeamOperator
        B_inv_freq = [BlockDiagonalOperator(inverse_beam_operators[f]) for f in range(N_freq)]
        r_list = [B_inv_freq[f].mv(d_list[f]) for f in range(N_freq)]

        return ListToStokesOperator(axis=0, in_structure=self.in_structure()).mv(r_list)
