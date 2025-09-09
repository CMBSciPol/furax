import equinox
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact, PyTree
from scipy.sparse import load_npz

from furax import AbstractLinearOperator, symmetric


@symmetric
class BeamOperatorMapspace(AbstractLinearOperator):
    """
    BeamOperatorMapspace applies a beam to a map in map space.

    Args:
        _in_structure (PyTree[jax.ShapeDtypeStruct]): Input structure of the operator.
        path_to_file (str): Path to the file containing the sparse beam matrix in .npz format.
        sparse_Beam_matrix (Float[Array, 'a b']): Sparse matrix representing the beam.
        _N_neighbours (int): Number of non-zero entries per row in the sparse matrix.
    Returns:
        PyTree[Inexact[Array, ' _b']]: Output map after applying the beam.
    """
    
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    sparse_Beam_matrix: Float[Array, 'a b'] = equinox.field(static=True)
    _N_neighbours: int = equinox.field(static=True)

    def __init__(self, in_structure: PyTree[jax.ShapeDtypeStruct], path_to_file: str) -> None:
        self._in_structure = in_structure
        self.sparse_Beam_matrix = load_npz(path_to_file)
        self._N_neighbours = int(jnp.diff(self.sparse_Beam_matrix.indptr)[0])

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]: 
        def apply_sparse_matrix(stokes_comp):
            # here comes in one Stokes parameter with shape [freq, npix]

            def dot_product(mp):
                # Calculate the dot product of the sparse matrix with one map mp

                # I could take it out of here if it's the same beam for all frequency channels 
                # and all Stokes parameters
                indices = jnp.array(self.sparse_Beam_matrix.indices)
                data = jnp.array(self.sparse_Beam_matrix.data)
                n_rows = self.sparse_Beam_matrix.shape[0]

                # self._N_neighbours could be calculated here if we have different 
                # number of neighbours per frequency channel 
                all_indices = indices.reshape(n_rows, self._N_neighbours)
                all_data = data.reshape(n_rows, self._N_neighbours)

                def explicit_dot_product(row_data, row_indices):
                    # here I operate on each pixel of map mp individually
                    return jnp.sum(row_data * mp[row_indices])

                return jax.vmap(explicit_dot_product)(all_data, all_indices) 

            return jax.vmap(dot_product)(jnp.atleast_2d(stokes_comp))
    
        return jax.tree.map(apply_sparse_matrix, x)
    
    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure
    