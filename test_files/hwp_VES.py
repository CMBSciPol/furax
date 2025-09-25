import jax.numpy as jnp
import numpy as np
import equinox
from jaxtyping import Float, Array
import equinox
import jax
import numpy as np
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import Float, PyTree, Inexact
import jax.numpy as jnp
from furax import AbstractLinearOperator, diagonal, square
from furax.core.rules import AbstractBinaryRule
from furax.obs.stokes import (
    StokesPyTreeType,
    ValidStokesType,
    StokesQU, 
    StokesIQU,
    StokesIQUV
)
from furax.obs.operators._qu_rotations import QURotationOperator, QURotationTransposeOperator
from furax.obs.landscapes import Stokes



# Pre-compute constants
c = 299792458.0  # m/s
GHz = 1e9
deg = jnp.pi / 180

def delta(nu, theta, n, nO=3.05):
    """Compute phase difference - vectorized for efficiency"""
    return 2 * jnp.pi * nu * (nO - n) * theta / c

def compute_effective_index(angleIncidence, chi, nE=3.38, nO=3.05):
    """Pre-compute effective refractive index"""
    sin_inc_sq = jnp.sin(angleIncidence)**2
    cos_chi_sq = jnp.cos(chi)**2
    return nE * jnp.sqrt(1 + (nE**-2 - nO**-2) * sin_inc_sq * cos_chi_sq)

def HWP_VES_efficient(nu, theta, n_eff, nO=3.05):
    """Efficient HWP Mueller matrix computation"""
    d = delta(nu, theta, n_eff, nO)
    cos_d = jnp.cos(d)
    sin_d = jnp.sin(d)
    
    # Only compute non-zero elements directly
    return jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, cos_d, -sin_d],
        [0.0, 0.0, sin_d, cos_d]
    ])

def rotation_matrix_2d(angle):
    """Efficient 2D rotation matrix for QU components"""
    cos_2a = jnp.cos(2 * angle)
    sin_2a = jnp.sin(2 * angle)
    return jnp.array([
        [cos_2a, sin_2a],
        [-sin_2a, cos_2a]
    ])

@diagonal
class MixedStokesOperator(AbstractLinearOperator):
    """Optimized Half-wave plate operator"""
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    frequency: float
    # Pre-computed values - stored as regular fields, not static
    _n_eff_layer1: jax.Array
    _n_eff_layer2: jax.Array
    _thickness: float
    _alpha_2: float
    _epsilon: float
    _phi: float
    # Pre-computed Mueller matrix elements for QU subspace
    _m11_m22: jax.Array
    _m12_m21: jax.Array

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesType = 'IQU',
        *,
        frequency: float,
        angleIncidence: float,
        epsilon: float,
        phi: float,
        thickness: float,
        alpha_2: float,
        angles: Float[Array, '...'] | None = None,
    ) -> AbstractLinearOperator:
        # Ensure dtype is float64
        dtype = np.float64
        in_structure = Stokes.class_for(stokes).structure_for(shape, dtype)
        
        # Pre-compute effective indices
        n_eff_1 = compute_effective_index(angleIncidence * deg, 0.0)
        n_eff_2 = compute_effective_index(angleIncidence * deg, alpha_2)
        
        # Pre-compute the complete Mueller matrix transformation for QU
        nO = 3.05
        # Layer 1
        HWP1 = HWP_VES_efficient(frequency * GHz, thickness, n_eff_1, nO)
        # Layer 2 with rotation
        HWP2_base = HWP_VES_efficient(frequency * GHz, thickness, n_eff_2, nO)
        
        # Full Mueller matrix product (pre-computed)
        Mueller_full = HWP1 @ rotation_matrix_mueller(alpha_2) @ HWP2_base @ rotation_matrix_mueller(-alpha_2) @ HWP1
        
        # Apply sign flips and extract QU submatrix
        Mueller_qu = Mueller_full[1:3, 1:3]
        Mueller_qu = Mueller_qu.at[1, :].multiply(-1)  # Q -> -Q for second row
        Mueller_qu = Mueller_qu.at[:, 1].multiply(-1)  # U -> -U for second column
        
        # Pre-compute the final transformation coefficients
        m11_m22 = Mueller_qu[0, 0] - Mueller_qu[1, 1]
        m12_m21 = Mueller_qu[0, 1] + Mueller_qu[1, 0]
        
        hwp = cls(
            _in_structure=in_structure,
            frequency=frequency,
            _n_eff_layer1=n_eff_1,
            _n_eff_layer2=n_eff_2,
            _thickness=thickness,
            _alpha_2=alpha_2,
            _epsilon=epsilon,
            _phi=phi,
            _m11_m22=m11_m22,
            _m12_m21=m12_m21,
        )
        
        if angles is None:
            return hwp
        
        # Apply final rotation if needed
        rot = QURotationOperator(angles + phi, in_structure)
        return rot.T @ hwp @ rot

    # ============================================================================
    # FIXED mv() METHOD - REMOVE ALL EXPLICIT DTYPE CASTING
    # ============================================================================
    def mv(self, x: StokesPyTreeType) -> Stokes:
        """Ultra-optimized matrix-vector product - JAX-compatible"""
        # Direct computation using pre-computed values
        q_new = 0.5 * (self._m11_m22 * x.q + self._m12_m21 * x.u)
        u_new = 0.5 * (self._m12_m21 * x.q + self._m11_m22 * x.u)
        
        # REMOVED ALL jnp.asarray(x, dtype=np.float64) calls!
        # Let JAX handle dtype promotion automatically
        
        if isinstance(x, StokesQU):
            return StokesQU(q_new, u_new)
        elif isinstance(x, StokesIQU):
            # No explicit dtype casting - just direct computation
            return StokesIQU(x.i, self._epsilon * q_new, self._epsilon * u_new)
        elif isinstance(x, StokesIQUV):
            return StokesIQUV(x.i, q_new, u_new, x.v)
        else:
            raise NotImplementedError(f"Stokes type {type(x)} not supported")

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure
    
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

# ============================================================================
# FIXED rotation_matrix_mueller - REMOVE EXPLICIT DTYPE
# ============================================================================
def rotation_matrix_mueller(angle):
    """4x4 Mueller matrix for rotation"""
    cos_2a = jnp.cos(2 * angle)
    sin_2a = jnp.sin(2 * angle)
    return jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos_2a, sin_2a, 0.0],
        [0.0, -sin_2a, cos_2a, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])  # REMOVED: dtype=np.float64


class StokesToListOperator(AbstractLinearOperator):
    """Operator that converts StokesIQU[Array[n,m]] to [StokesIQU[Array[m]], ...].
    Index notation: x[stokes,(freq,pix)] -> x[freq,stokes,(pix)]
    
    Attributes:
        axis: The axis along which the leaves were originally stacked.
        in_structure: The in_structure of the pytree to be unstacked.
        num_elements: Number of elements along the splitting axis (static).
    """
    axis: int = equinox.field(static=True)  # Make axis static
    num_elements: int = equinox.field(static=True)  # Store size statically
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    
    def __init__(
        self,
        axis: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        num_elements: int | None = None,
    ):
        self.axis = axis
        self._in_structure = in_structure
        
        # If num_elements not provided, try to infer from in_structure
        if num_elements is None:
            if hasattr(in_structure, 'i') and hasattr(in_structure.i, 'shape'):
                num_elements = in_structure.i.shape[axis]
            else:
                raise ValueError("num_elements must be provided if it cannot be inferred from in_structure")
        
        self.num_elements = num_elements
        # assert isinstance(in_structure, StokesIQU), 'Wrong format: the input should be a single StokesIQU tree'
    
    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure
    
    def mv(self, x: Inexact[Array, '...']) -> PyTree[Inexact[Array, '...']]:
        return [
            StokesIQU(
                i=jnp.take(x.i, f, axis=self.axis),
                q=jnp.take(x.q, f, axis=self.axis),
                u=jnp.take(x.u, f, axis=self.axis),
            )
            for f in range(self.num_elements)
        ]
    

class ListToStokesOperator(AbstractLinearOperator):
    """Operator that converts [StokesIQU[Array[m]], ... ] to StokesIQU[Array[n,m]].
    Index notation: x[freq,stokes,(pix)] -> x[stokes,(freq,pix)]
    
    Attributes:
        axis: The axis along which the leaves are stacked.
        in_structure: The in_structure of the pytree to be stacked.
    """
    axis: int = equinox.field(static=True)  # Make axis static
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    
    def __init__(
        self,
        axis: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        self.axis = axis
        self._in_structure = in_structure
        assert isinstance(in_structure, list) and isinstance(in_structure[0], StokesIQU), \
            'Wrong format: the input should be a list of StokesIQU trees'
    
    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure
    
    def mv(self, x: PyTree[Inexact[Array, '...']]) -> Inexact[Array, '...']:
        # Extract all i, q, u components
        i_arrays = [x_i.i for x_i in x]
        q_arrays = [x_i.q for x_i in x]
        u_arrays = [x_i.u for x_i in x]
        
        return StokesIQU(
            i=jnp.stack(i_arrays, axis=self.axis),
            q=jnp.stack(q_arrays, axis=self.axis),
            u=jnp.stack(u_arrays, axis=self.axis),
        )