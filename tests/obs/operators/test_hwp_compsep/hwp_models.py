import equinox
import jax
import numpy as np
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import Float, PyTree, Inexact
import jax.numpy as jnp
from furax import AbstractLinearOperator, diagonal, square
from furax.core.rules import AbstractBinaryRule
import tools as tl 
from furax.obs.stokes import (
    Stokes,
    StokesI,
    StokesIQU,
    StokesIQUV,
    StokesPyTreeType,
    StokesQU,
    ValidStokesType,
)
from furax.obs.operators._qu_rotations import QURotationOperator, QURotationTransposeOperator

import transfer_matrixJAX as tm

# Material definitions (unchanged)
sapphire = tm.material(3.05, 3.38, 2.3e-4, 1.25e-4, 'Sapphire', materialType='uniaxial')
duroid   = tm.material(1.41, 1.41, 1.2e-3, 1.2e-3, 'RT Duroid', materialType='isotropic')
mullite = tm.material(2.52, 2.52, 0.0121, 0.0121, 'Mullite', materialType='isotropic')
epoteck = tm.material(1.7, 1.7, 0., 0., 'Epoteck', materialType='isotropic')

thicknesses   = [0.394*tm.mm, 0.04*tm.mm, 0.212*tm.mm, 3.75*tm.mm, 3.75*tm.mm, 3.75*tm.mm, 0.212*tm.mm, 0.04*tm.mm, 0.394*tm.mm]
thicknesses_HF = [0.183*tm.mm, 0.04*tm.mm, 0.097*tm.mm, 1.60*tm.mm, 1.60*tm.mm, 1.60*tm.mm, 0.097*tm.mm, 0.04*tm.mm, 0.183*tm.mm]
materials   = [duroid, epoteck, mullite, sapphire, sapphire, sapphire, mullite, epoteck, duroid]
angles_MF      = [0.0, 0.0, 0.0, 0.0, 54.0*tm.deg, 0.0, 0.0, 0.0, 0.0]
angles_HF    = [0.0, 0.0, 0.0, 0.0, 57.0*tm.deg, 0.0, 0.0, 0.0, 0.0]
hwp_stack   = tm.Stack(thicknesses, materials, angles_MF)
hwp_stack_HF = tm.Stack(thicknesses_HF, materials, angles_HF)


c = 299792458.0  # m/s
GHz = 1e9 
deg = jnp.pi/180.0  
angleIncidence = 5.0


def get_delta(nu, theta, n, nO=3.05):
    """Compute phase difference with better precision"""
    return 2 * jnp.pi * nu * (nO - n) * theta / c

def compute_effective_index(angleIncidence, chi, nE=3.38, nO=3.05):
    """Compute effective refractive index"""
    sin_inc_sq = jnp.sin(angleIncidence)**2
    cos_chi_sq = jnp.cos(chi)**2
    return nE * jnp.sqrt(1 + (nE**-2 - nO**-2) * sin_inc_sq * cos_chi_sq)

def HWP_Stack(nu, theta, angleIncidence, chi, nE=3.38, nO=3.05):
    n = compute_effective_index(angleIncidence, chi, nE, nO)
    d = get_delta(nu, theta, n, nO)
    cos_d = jnp.cos(d)
    sin_d = jnp.sin(d)
    
    return jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, cos_d, -sin_d],
        [0.0, 0.0, sin_d, cos_d]
    ])

def rotation_matrix_mueller(angle):
    """4x4 Mueller matrix for rotation"""
    cos_2a = jnp.cos(2 * angle)
    sin_2a = jnp.sin(2 * angle)
    return jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos_2a, -sin_2a, 0.0],
        [0.0, sin_2a, cos_2a, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def m11m22(alpha,theta,freq):
    return(m11(alpha,theta,freq)-m22(alpha,theta,freq))

def m12m21(alpha,theta,freq):
    return(m12(alpha,theta,freq)+m21(alpha,theta,freq))

def m12m21b(alpha,theta,freq):
    return(-(m12(alpha,theta,freq)+m21(alpha,theta,freq)))

def m11(alpha,theta,nu):
    delta = get_delta(nu, theta, n=3.38, nO=3.05)
    # alpha *= deg
    return(jnp.cos(2*alpha)**2+jnp.cos(delta)*jnp.sin(2*alpha)**2)

def m22(alpha,theta,nu):
    delta = get_delta(nu, theta, n=3.38, nO=3.05)
    # alpha *= deg
    return(jnp.cos(delta)**2*jnp.sin(2*alpha)**2+jnp.cos(delta)**3*jnp.cos(2*alpha)**2-2*jnp.sin(delta)**2*jnp.cos(delta)*jnp.cos(2*alpha)-jnp.sin(delta)**2*jnp.cos(delta))

def m12(alpha,theta,nu):
    delta = get_delta(nu, theta, n=3.38, nO=3.05)
    # alpha *= deg
    return(jnp.sin(2*alpha)*jnp.cos(2*alpha)*(jnp.cos(delta)**2-jnp.cos(delta))-jnp.sin(delta)**2*jnp.sin(2*alpha))

def m21(alpha,theta,freq):
    return(m12(alpha,theta,freq))


@square
class TMHWPOperator(AbstractLinearOperator):
    """Transfer matrix HWP operator with better structure"""
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    frequency: float
    angleIncidence: float

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesType = 'IQU',
        *,
        frequency: float,
        angleIncidence: float,
        angles: Float[Array, '...'] | None = None,
    ) -> AbstractLinearOperator:
        in_structure = Stokes.class_for(stokes).structure_for(shape, dtype)
        hwp = cls(in_structure, frequency, angleIncidence)
        
        if angles is None:
            return hwp
        
        rot = QURotationOperator(angles, in_structure)
        return rot.T @ hwp @ rot

    def mv(self, x: StokesPyTreeType) -> Stokes:
        Mueller_ = jnp.array(tm.Mueller(hwp_stack, self.frequency*GHz, self.angleIncidence*deg, 0., reflected=False))
        Mueller_iqu = Mueller_[:-1, :-1] 
        
        # Matrix-vector multiplication
        i = Mueller_iqu[0,0] * x.i + Mueller_iqu[0,1] * x.q + Mueller_iqu[0,2] * x.u
        q = Mueller_iqu[1,0] * x.i + Mueller_iqu[1,1] * x.q + Mueller_iqu[1,2] * x.u
        u = Mueller_iqu[2,0] * x.i + Mueller_iqu[2,1] * x.q + Mueller_iqu[2,2] * x.u
        
        if isinstance(x, StokesIQU):
            return StokesIQU(i, q, u)
        elif isinstance(x, StokesIQUV):
            # For IQUV, compute V component
            v = Mueller_[3,0] * x.i + Mueller_[3,1] * x.q + Mueller_[3,2] * x.u + Mueller_[3,3] * x.v
            return StokesIQUV(i, q, u, v)
        
        raise NotImplementedError(f"Stokes type {type(x)} not supported")

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

@square  
class TMHWP_HFOperator(AbstractLinearOperator):
    """HF version of transfer matrix HWP operator"""
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    frequency: float
    angleIncidence: float

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesType = 'IQU',
        *,
        frequency: float,
        angleIncidence: float,
        angles: Float[Array, '...'] | None = None,
    ) -> AbstractLinearOperator:
        in_structure = Stokes.class_for(stokes).structure_for(shape, dtype)
        hwp = cls(in_structure, frequency, angleIncidence)
        
        if angles is None:
            return hwp
            
        rot = QURotationOperator(angles, in_structure)
        return rot.T @ hwp @ rot

    def mv(self, x: StokesPyTreeType) -> Stokes:
        Mueller_ = jnp.array(tm.Mueller(hwp_stack_HF, self.frequency*GHz, self.angleIncidence*deg, 0., reflected=False))
        Mueller_iqu = Mueller_[:-1, :-1]
        
        i = Mueller_iqu[0,0] * x.i + Mueller_iqu[0,1] * x.q + Mueller_iqu[0,2] * x.u
        q = Mueller_iqu[1,0] * x.i + Mueller_iqu[1,1] * x.q + Mueller_iqu[1,2] * x.u
        u = Mueller_iqu[2,0] * x.i + Mueller_iqu[2,1] * x.q + Mueller_iqu[2,2] * x.u
        
        if isinstance(x, StokesIQU):
            return StokesIQU(i, q, u)
        elif isinstance(x, StokesIQUV):
            v = Mueller_[3,0] * x.i + Mueller_[3,1] * x.q + Mueller_[3,2] * x.u + Mueller_[3,3] * x.v
            return StokesIQUV(i, q, u, v)
            
        raise NotImplementedError(f"Stokes type {type(x)} not supported")

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


@diagonal
class HWPStackOperator(AbstractLinearOperator):
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    frequency: float
    _Mueller_qu: jax.Array  # Fixed: was None
    _thickness: float
    _alpha_2: float
    _epsilon: float
    _phi: float
    
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
        
        # Layer 1
        HWP1 = HWP_Stack(frequency*GHz, thickness, angleIncidence*deg, chi=0.)
        HWP2_base = HWP_Stack(frequency*GHz, thickness, angleIncidence*deg, chi=alpha_2)
        
        # Full Mueller matrix product
        Mueller_full    = HWP1 @ rotation_matrix_mueller(alpha_2).T @ HWP2_base @ rotation_matrix_mueller(alpha_2) @ HWP1
        Mueller_qu      = Mueller_full.at[2, :].multiply(-1)  # Q -> -Q for second row
        Mueller_qu      = Mueller_qu.at[:, 2].multiply(-1)  # U -> -U for second column
        Mueller_qu      = Mueller_qu[:-1, :-1]
        
        hwp = cls(
            _in_structure=in_structure,
            frequency=frequency,
            _thickness=thickness,
            _alpha_2=alpha_2,
            _epsilon=epsilon,
            _phi=phi,
            _Mueller_qu=Mueller_qu  
        )
        
        if angles is None:
            return hwp
        
        rot = QURotationOperator(angles + phi, in_structure)
        return rot.T @ hwp @ rot


    def mv(self, x: StokesPyTreeType) -> Stokes:
        Mueller_qu = self._Mueller_qu
        
        # Apply Mueller matrix transformation
        i_new = Mueller_qu[0, 0] * x.i + Mueller_qu[0, 1] * x.q + Mueller_qu[0, 2] * x.u
        q_new = Mueller_qu[1, 0] * x.i + Mueller_qu[1, 1] * x.q + Mueller_qu[1, 2] * x.u
        u_new = Mueller_qu[2, 0] * x.i + Mueller_qu[2, 1] * x.q + Mueller_qu[2, 2] * x.u
        
        # Return appropriate Stokes type
        if isinstance(x, StokesQU):
            return StokesQU(q_new, u_new)
        elif isinstance(x, StokesIQU):
            return StokesIQU(i_new, self._epsilon * q_new, self._epsilon * u_new)
        elif isinstance(x, StokesIQUV):
            return StokesIQUV(i_new, q_new, u_new, x.v) 
        else:
            raise NotImplementedError(f"Stokes type {type(x)} not supported")
    
    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure
    
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


@diagonal
class HWPStackBPOperator(AbstractLinearOperator):
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    frequency: float
    _Mueller_qu: jax.Array  
    _thickness: float
    _alpha_2: float
    _epsilon: float
    _phi: float
    _nfreq: int
    
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
        NFREQ : int,
        angles: Float[Array, '...'] | None = None,
    ) -> AbstractLinearOperator:
        # Ensure dtype is float64
        dtype = np.float64
        in_structure = Stokes.class_for(stokes).structure_for(shape, dtype)

        nu_array, weights = tl.get_bp_uKCMB(frequency,NFREQ)
        m_dust = tl.get_Adust_fgbuster(nu_array)
        norm = jnp.sum(weights * m_dust)
        Mueller_avg = jnp.zeros((4, 4))
        
        # Layer 1
        HWP1 = HWP_Stack(frequency*GHz, thickness, angleIncidence*deg, chi=0.)
        HWP2_base = HWP_Stack(frequency*GHz, thickness, angleIncidence*deg, chi=alpha_2)
        
        # Full Mueller matrix product
        for nu, w, s in zip(nu_array, weights, m_dust):
            Mueller_full    = HWP1 @ rotation_matrix_mueller(alpha_2).T @ HWP2_base @ rotation_matrix_mueller(alpha_2) @ HWP1
            Mueller_full      = Mueller_full.at[2, :].multiply(-1)  # Q -> -Q for second row
            Mueller_full      = Mueller_full.at[:, 2].multiply(-1)  # U -> -U for second column
            Mueller_avg += w * s * Mueller_full / norm
            Mueller_qu      = Mueller_avg[:-1, :-1]
            
        hwp = cls(
            _in_structure=in_structure,
            frequency=frequency,
            _thickness=thickness,
            _alpha_2=alpha_2,
            _epsilon=epsilon,
            _phi=phi,
            _Mueller_qu=Mueller_qu,
            _nfreq = NFREQ
        )
        
        if angles is None:
            return hwp
        
        rot = QURotationOperator(angles + phi, in_structure)
        return rot.T @ hwp @ rot


    def mv(self, x: StokesPyTreeType) -> Stokes:
        Mueller_qu = self._Mueller_qu
        
        # Apply Mueller matrix transformation
        i_new = Mueller_qu[0, 0] * x.i + Mueller_qu[0, 1] * x.q + Mueller_qu[0, 2] * x.u
        q_new = Mueller_qu[1, 0] * x.i + Mueller_qu[1, 1] * x.q + Mueller_qu[1, 2] * x.u
        u_new = Mueller_qu[2, 0] * x.i + Mueller_qu[2, 1] * x.q + Mueller_qu[2, 2] * x.u
        
        # Return appropriate Stokes type
        if isinstance(x, StokesQU):
            return StokesQU(q_new, u_new)
        elif isinstance(x, StokesIQU):
            return StokesIQU(i_new, self._epsilon * q_new, self._epsilon * u_new)
        elif isinstance(x, StokesIQUV):
            return StokesIQUV(i_new, q_new, u_new, x.v)  # Fixed: use i_new
        else:
            raise NotImplementedError(f"Stokes type {type(x)} not supported")
    
    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure
    
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


@diagonal
class MixedStokesOperator(AbstractLinearOperator):
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    frequency: float
    
    _thickness: float
    _alpha_2: float
    _epsilon: float
    _phi: float
    
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
            
        # # Layer 1
        HWP1        = HWP_Stack(frequency*GHz, thickness, angleIncidence*deg, chi=0.)
        HWP2_base   = HWP_Stack(frequency*GHz, thickness, angleIncidence*deg, chi=alpha_2)
        
        # Full Mueller matrix product
        Mueller_full    = HWP1 @ rotation_matrix_mueller(alpha_2).T @ HWP2_base @ rotation_matrix_mueller(alpha_2) @ HWP1
        Mueller_qu      = Mueller_full.at[2, :].multiply(-1)  # Q -> -Q for second row
        Mueller_qu      = Mueller_qu.at[:, 2].multiply(-1)  # U -> -U for second column
        Mueller_qu      = Mueller_qu[:-1, :-1]
        
        # Pre-compute the final transformation coefficients
        m11_m22     = Mueller_qu[1 , 1] - Mueller_qu[2 , 2]
        # m11_m220    = Mueller_qu[1 , 1] + Mueller_qu[2 , 2] 
        # m11_m22   = m11m22(alpha_2,thickness,frequency * GHz) #
        m12_m21     = Mueller_qu[1 , 2] + Mueller_qu[2 , 1] #m12m21(alpha_2,thickness,frequency * GHz) #
        # m12_m21 = m12m21(alpha_2,thickness,frequency * GHz) #

        hwp = cls(
            _in_structure=in_structure,
            frequency=frequency,
            _thickness=thickness,
            _alpha_2= alpha_2,
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

        
        q_new = 0.5*(self._m11_m22 * x.q    - self._m12_m21 * x.u)
        u_new = 0.5*(-self._m12_m21 * x.q   - self._m11_m22 * x.u)
        
        
        if isinstance(x, StokesQU):
            return StokesQU(q_new, u_new)
        elif isinstance(x, StokesIQU):
            return StokesIQU(x.i, self._epsilon * q_new, self._epsilon * u_new)
            # return StokesIQU(x.i, x.q, x.u)
        elif isinstance(x, StokesIQUV):
            return StokesIQUV(x.i, q_new, u_new, x.v)
        else:
            raise NotImplementedError(f"Stokes type {type(x)} not supported")

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure
    
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

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
    
