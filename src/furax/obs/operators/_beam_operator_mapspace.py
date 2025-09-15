import equinox
import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact, PyTree

from furax import AbstractLinearOperator, symmetric


@symmetric
class BeamOperatorMapspace(AbstractLinearOperator):
    """
    BeamOperatorMapspace applies a beam to a map in map space.

    Args:
        _in_structure (PyTree[jax.ShapeDtypeStruct]): Input structure of the operator.
        _indices (Array): Indices of the sparse matrix in CSR format.
        _data (Array): Data of the sparse matrix in CSR format.
        _N_neighbours (int): Number of non-zero entries per row in the sparse matrix.
        _n_rows (int): Number of rows in the sparse matrix.
    Returns:
        PyTree[Inexact[Array, ' _b']]: Output map after applying the beam.
    """

    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    _N_neighbours: int = equinox.field(static=False)
    _indices: Array = equinox.field(static=False)
    _data: Array = equinox.field(static=False)
    _n_rows: int = equinox.field(static=False)

    def __init__(self, in_structure: PyTree[jax.ShapeDtypeStruct], indices: Array, data: Array, N_neighbours: int, n_rows: int) -> None:
        self._in_structure = in_structure
        self._indices = indices
        self._data = data
        self._N_neighbours = N_neighbours
        self._n_rows = n_rows

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]: 
        def apply_sparse_matrix(stokes_comp):
            # this function takes in one Stoke's parameter with shape [freq, npix]
            
            def dot_product(mp):
                # Calculate the dot product of the sparse matrix with one frequency map

                # I could take it out of here if it's the same beam for all frequency channels 
                # and all Stokes parameters. self._N_neighbours could also be calculated here 
                # if we have different number of neighbours per frequency channel 
                all_indices = self._indices.reshape(self._n_rows, self._N_neighbours)
                all_data = self._data.reshape(self._n_rows, self._N_neighbours)

                return jnp.sum(all_data * mp[all_indices], axis=1)

            return jax.vmap(dot_product)(jnp.atleast_2d(stokes_comp))
    
        return jax.tree.map(apply_sparse_matrix, x)
    
    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure
    