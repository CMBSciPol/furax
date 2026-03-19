import sys
from abc import abstractmethod
from dataclasses import field
from typing import Any

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

import jax
import jax.numpy as jnp
from astropy.cosmology import Planck15
from jaxtyping import Array, Float, Inexact, Int, PyTree
from scipy import constants

from furax import AbstractLinearOperator, BlockRowOperator, BroadcastDiagonalOperator, diagonal

_H_OVER_K_GHZ = constants.h * 1e9 / constants.k
_T_CMB = Planck15.Tcmb(0).value

__all__ = [
    'AbstractSEDOperator',
    'CMBOperator',
    'DustOperator',
    'SynchrotronOperator',
    'MixingMatrixOperator',
]


def K_RK_2_K_CMB(nu: Array | float) -> Array:
    """
    Convert Rayleigh-Jeans brightness temperature to CMB temperature.

    .. math::
        T_{CMB} = \frac{(e^{\frac{h \nu}{k T_{CMB}}} - 1)^2}{(e^{\frac{h \nu}{k T_{CMB}}})
        \\left( \frac{h \nu}{k T_{CMB}} \right)^2}

    Args:
        nu (Array | float): Frequency in GHz.

    Returns:
        Array: Conversion factor from Rayleigh-Jeans to CMB temperature.

    Example:
        >>> nu = jnp.array([30, 40, 100])
        >>> conversion = K_RK_2_K_CMB(nu)
        >>> print(conversion)
    """
    res = jnp.expm1(_H_OVER_K_GHZ * nu / _T_CMB) ** 2 / (
        jnp.exp(_H_OVER_K_GHZ * nu / _T_CMB) * (_H_OVER_K_GHZ * nu / _T_CMB) ** 2
    )
    return res  # type: ignore [no-any-return]


class AbstractSEDOperator(BroadcastDiagonalOperator):
    """Abstract base class for Spectral Energy Distribution (SED) operators.

    SED operators model how astrophysical components emit radiation across
    frequencies. They broadcast sky maps to multiple frequency channels.

    Subclasses must implement the ``sed()`` method to define the spectral
    energy distribution.

    Attributes:
        frequencies: Array of observation frequencies.
        frequency0: Reference frequency for the SED normalization.
    """

    frequencies: Float[Array, ' a']
    frequency0: float = field(metadata={'static': True})

    def __init__(
        self,
        frequencies: Float[Array, '...'],
        *,
        frequency0: float = 100e9,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        input_shape = self._get_input_shape(in_structure)
        frequencies = frequencies.reshape((len(frequencies),) + tuple(1 for _ in input_shape))
        object.__setattr__(self, 'frequencies', frequencies)
        object.__setattr__(self, 'frequency0', frequency0)
        super().__init__(self.sed(), axis_destination=-1, in_structure=in_structure)

    @staticmethod
    def _get_input_shape(in_structure: PyTree[jax.ShapeDtypeStruct]) -> tuple[int, ...]:
        """
        Determine the shape of the input leaves in the PyTree.

        Args:
            in_structure (PyTree): The PyTree structure.

        Returns:
            tuple[int, ...]: The common shape of the leaves.

        Raises:
            ValueError: If the shapes of the leaves are not consistent.
        """
        input_shapes = set(leaf.shape for leaf in jax.tree.leaves(in_structure))
        if len(input_shapes) != 1:
            raise ValueError(f'the leaves of the input do not have the same shape: {in_structure}')
        return input_shapes.pop()  # type: ignore[no-any-return]

    @abstractmethod
    def sed(self) -> Float[Array, '...']:
        """
        Define the spectral energy distribution transformation.

        Returns:
            Float[Array, '...']: The transformed SED.
        """
        ...

    @staticmethod
    def _get_at(
        values: Float[Array, '...'], indices: Int[Array, '...'] | None
    ) -> Float[Array, '...']:
        """
        Retrieve values at specified indices, or return all values if indices are None.

        Args:
            values (Array): Input array.
            indices (Array | None): Indices to retrieve values from.

        Returns:
            Array: Subset of values or the entire array.
        """
        if indices is None:
            return values
        return values[..., indices]


class CMBOperator(AbstractSEDOperator):
    """Operator for Cosmic Microwave Background (CMB) spectral energy distribution.

    The CMB has a blackbody spectrum at T_CMB ~ 2.725 K. In K_CMB units, the
    SED is constant (unity) across frequencies. In K_RJ units, a frequency-dependent
    conversion factor is applied.

    Attributes:
        frequencies: Observation frequencies [GHz].
        units: Output units ('K_CMB' or 'K_RJ').
        factor: Unit conversion factor.

    Example:
        >>> nu = jnp.array([30, 40, 100])  # GHz
        >>> cmb_op = CMBOperator(frequencies=nu, in_structure=landscape.structure)
        >>> tod = cmb_op(sky_map)  # Broadcasts CMB map to all frequencies
    """

    factor: Float[Array, '...'] | float
    units: str = field(metadata={'static': True})

    def __init__(
        self,
        frequencies: Float[Array, '...'],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        units: str = 'K_CMB',
    ) -> None:
        factor: Float[Array, ...] | float
        if units == 'K_CMB':
            factor = 1.0
        elif units == 'K_RJ':
            factor = K_RK_2_K_CMB(frequencies)
        else:
            raise ValueError(f"Unknown units: {units}. Expected 'K_CMB' or 'K_RJ'.")
        object.__setattr__(self, 'factor', factor)
        object.__setattr__(self, 'units', units)
        super().__init__(frequencies, in_structure=in_structure)

    def sed(self) -> Float[Array, '...']:
        """
        Compute the spectral energy distribution for the CMB.

        Returns:
            Float[Array, '...']: The SED for the CMB.
        """
        return jnp.ones_like(self.frequencies) / jnp.expand_dims(self.factor, axis=-1)


class DustOperator(AbstractSEDOperator):
    """Operator for thermal dust spectral energy distribution.

    Models dust emission as a modified blackbody: a power law times a Planck
    function. The SED is: (nu/nu0)^(1+beta) * B(nu,T)/B(nu0,T), where B is
    the Planck function.

    Supports spatially varying spectral parameters via patch indices.

    Attributes:
        frequencies: Observation frequencies [GHz].
        frequency0: Reference frequency [GHz].
        temperature: Dust temperature [K].
        beta: Spectral index (typically ~1.5).
        units: Output units ('K_CMB' or 'K_RJ').

    Example:
        >>> nu = jnp.array([100, 143, 217, 353])  # GHz
        >>> dust_op = DustOperator(
        ...     frequencies=nu, frequency0=353, beta=1.54, temperature=20.0,
        ...     in_structure=landscape.structure
        ... )
        >>> tod = dust_op(dust_map)
    """

    temperature: Float[Array, '...']
    temperature_patch_indices: Int[Array, '...'] | None
    beta: Float[Array, '...']
    beta_patch_indices: Int[Array, '...'] | None
    factor: Float[Array, '...'] | float
    units: str = field(metadata={'static': True})

    def __init__(
        self,
        frequencies: Float[Array, '...'],
        *,
        frequency0: float = 100,
        temperature: float | Float[Array, '...'],
        units: str = 'K_CMB',
        temperature_patch_indices: Int[Array, '...'] | None = None,
        beta: float | Float[Array, '...'],
        beta_patch_indices: Int[Array, '...'] | None = None,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        factor: Float[Array, ...] | float
        if units == 'K_CMB':
            factor = K_RK_2_K_CMB(frequencies) / K_RK_2_K_CMB(frequency0)
        elif units == 'K_RJ':
            factor = 1.0
        else:
            raise ValueError(f"Unknown units: {units}. Expected 'K_CMB' or 'K_RJ'.")
        object.__setattr__(self, 'temperature', jnp.asarray(temperature))
        object.__setattr__(self, 'temperature_patch_indices', temperature_patch_indices)
        object.__setattr__(self, 'beta', jnp.asarray(beta))
        object.__setattr__(self, 'beta_patch_indices', beta_patch_indices)
        object.__setattr__(self, 'units', units)
        object.__setattr__(self, 'factor', factor)

        super().__init__(frequencies, frequency0=frequency0, in_structure=in_structure)

    def sed(self) -> Float[Array, '...']:
        t = self._get_at(
            jnp.expm1(self.frequency0 / self.temperature * _H_OVER_K_GHZ)
            / jnp.expm1(self.frequencies / self.temperature * _H_OVER_K_GHZ),
            self.temperature_patch_indices,
        )
        b = self._get_at(
            (self.frequencies / self.frequency0) ** (1 + self.beta), self.beta_patch_indices
        )
        sed = (t * b) * jnp.expand_dims(self.factor, axis=-1)
        return sed


class SynchrotronOperator(AbstractSEDOperator):
    """Operator for synchrotron spectral energy distribution.

    Models synchrotron emission as a power law: (nu/nu0)^beta, with optional
    spectral index running: (nu/nu0)^(beta + running * log(nu/nu_pivot)).

    Supports spatially varying spectral parameters via patch indices.

    Attributes:
        frequencies: Observation frequencies [GHz].
        frequency0: Reference frequency [GHz].
        beta_pl: Power-law spectral index (typically ~ -3).
        nu_pivot: Pivot frequency for running [GHz].
        running: Running of the spectral index.
        units: Output units ('K_CMB' or 'K_RJ').

    Example:
        >>> nu = jnp.array([30, 44, 70])  # GHz
        >>> sync_op = SynchrotronOperator(
        ...     frequencies=nu, frequency0=30, beta_pl=-3.0,
        ...     in_structure=landscape.structure
        ... )
        >>> tod = sync_op(synchrotron_map)
    """

    beta_pl: Float[Array, '...']
    beta_pl_patch_indices: Int[Array, '...'] | None
    nu_pivot: float = field(metadata={'static': True})
    running: float = field(metadata={'static': True})
    units: str = field(metadata={'static': True})
    factor: Float[Array, '...'] | float

    def __init__(
        self,
        frequencies: Float[Array, '...'],
        *,
        frequency0: float = 100,
        nu_pivot: float = 1.0,
        running: float = 0.0,
        units: str = 'K_CMB',
        beta_pl: float | Float[Array, '...'],
        beta_pl_patch_indices: Int[Array, '...'] | None = None,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        factor: Float[Array, ...] | float
        if units == 'K_CMB':
            factor = K_RK_2_K_CMB(frequencies) / K_RK_2_K_CMB(frequency0)
        elif units == 'K_RJ':
            factor = 1.0
        else:
            raise ValueError(f"Unknown units: {units}. Expected 'K_CMB' or 'K_RJ'.")
        object.__setattr__(self, 'beta_pl', jnp.asarray(beta_pl))
        object.__setattr__(self, 'beta_pl_patch_indices', beta_pl_patch_indices)
        object.__setattr__(self, 'nu_pivot', nu_pivot)
        object.__setattr__(self, 'running', running)
        object.__setattr__(self, 'units', units)
        object.__setattr__(self, 'factor', factor)
        super().__init__(frequencies, frequency0=frequency0, in_structure=in_structure)

    def sed(self) -> Float[Array, '...']:
        sed = self._get_at(
            (
                (self.frequencies / self.frequency0)
                ** (self.beta_pl + self.running * jnp.log(self.frequencies / self.nu_pivot))
            ),
            self.beta_pl_patch_indices,
        )

        sed = self._get_at(
            (self.frequencies / self.frequency0) ** self.beta_pl, self.beta_pl_patch_indices
        )
        sed *= jnp.expand_dims(self.factor, axis=-1)

        return sed


def MixingMatrixOperator(**blocks: AbstractSEDOperator) -> AbstractLinearOperator:
    """Constructs a mixing matrix operator from a set of SED operators.

    This function combines multiple spectral energy distribution (SED) operators
    into a single block row operator for use in linear models.

    Args:
        **blocks: Named SED operators to combine into the mixing matrix.

    Returns:
        BlockRowOperator: A reduced block row operator representing the mixing matrix.

    Example:
        >>> from furax.obs import CMBOperator, DustOperator,\
             SynchrotronOperator, MixingMatrixOperator
        >>> nu = jnp.array([30, 40, 100])  # Frequencies in GHz
        >>> in_structure = ...  # Define input structure (e.g., using HealpixLandscape)
        >>> sky_map = ...  # Define sky map
        >>> cmb = CMBOperator(nu, in_structure=in_structure)
        >>> dust = DustOperator(
        ...     nu,
        ...     frequency0=150.0,
        ...     temperature=20.0,
        ...     beta=1.54,
        ...     in_structure=in_structure
        ... )
        >>> synchrotron = SynchrotronOperator(
        ...     nu,
        ...     frequency0=20.0,
        ...     beta_pl=-3.0,
        ...     in_structure=in_structure
        ... )
        >>> A = MixingMatrixOperator(cmb=cmb, dust=dust, synchrotron=synchrotron)
        >>> d = A(sky_map)
    """
    return BlockRowOperator(blocks).reduce()


@deprecated('Should use a DiagonalOperator')
@diagonal
class NoiseDiagonalOperator(AbstractLinearOperator):
    """Constructs a diagonal noise operator.

    This operator applies a noise vector (in a PyTree structure) in an element‐wise
    multiplication to an input data PyTree.

    Args:
        vector: PyTree of arrays representing the noise values.
        in_structure: Input structure (PyTree[jax.ShapeDtypeStruct]) specifying the shape and dtype.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax.obs.landscapes import FrequencyLandscape
        >>> from furax.obs.operators import NoiseDiagonalOperator
        >>>
        >>> landscape = FrequencyLandscape(nside=64, frequencies=jnp.linspace(30, 300, 10))
        >>> noise_sample = landscape.normal(jax.random.key(0))  # small n
        >>> d = landscape.normal(jax.random.key(0))  # d
        >>> N = NoiseDiagonalOperator(noise_sample, in_structure=d.structure)
        >>> N.I(d).structure
        StokesIQU(i=ShapeDtypeStruct(shape=(10, 49152), dtype=float64),
                  q=ShapeDtypeStruct(shape=(10, 49152), dtype=float64),
                  u=ShapeDtypeStruct(shape=(10, 49152), dtype=float64))
    """

    vector: PyTree[Inexact[Array, '...']]

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return jax.tree.map(lambda v, leaf: v * leaf, self.vector, x)

    def inverse(self) -> AbstractLinearOperator:
        return NoiseDiagonalOperator(vector=1 / self.vector, in_structure=self.in_structure)

    def as_matrix(self) -> Any:
        return jax.tree.map(lambda x: jnp.diag(x.flatten()), self.vector)
