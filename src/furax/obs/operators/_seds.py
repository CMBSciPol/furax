from abc import abstractmethod

import equinox
import jax
import jax.numpy as jnp
from astropy.cosmology import Planck15
from jaxtyping import Array, Float, Int, PyTree
from scipy import constants

from furax import AbstractLinearOperator, BlockRowOperator, BroadcastDiagonalOperator

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
    """
    Abstract base class for Spectral Energy Distribution (SED) operators.

    Args:
        frequencies (Array): Array of frequencies.
        frequency0 (float, optional): Reference frequency. Defaults to 100e9.
        in_structure (PyTree): Input structure describing the shape and dtype of the input.

    Attributes:
        frequencies (Array): Reshaped frequency array.
        frequency0 (float): Reference frequency.
        _in_structure (PyTree): Input structure.
    """

    frequencies: Float[Array, ' a']
    frequency0: float = equinox.field(static=True)
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        frequencies: Float[Array, '...'],
        *,
        frequency0: float = 100e9,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        input_shape = self._get_input_shape(in_structure)
        self.frequencies = frequencies.reshape((len(frequencies),) + tuple(1 for _ in input_shape))
        self.frequency0 = frequency0
        super().__init__(self.sed(), in_structure=in_structure)

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

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        """
        Returns the input structure of the operator.

        Returns:
            PyTree[jax.ShapeDtypeStruct]: The input structure.
        """
        return self._in_structure

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
    """
    Operator for Cosmic Microwave Background (CMB) spectral energy distribution.

    Args:
        frequencies (Array): Array of frequencies.
        in_structure (PyTree): Input structure describing the shape and dtype of the input.
        units (str, optional): Units for the operator ('K_CMB' or 'K_RJ'). Defaults to 'K_CMB'.

    Example:
        >>> from furax.operators.seds import CMBOperator
        >>> import jax.numpy as jnp
        >>> nu = jnp.array([30, 40, 100])  # Frequencies in GHz
        >>> in_structure = ...  # Define input structure (e.g., using HealpixLandscape)
        >>> sky_map = ...  # Define sky map
        >>> cmbOp = CMBOperator(
        ...     frequencies=nu,
        ...     in_structure=in_structure,
        ...     units='K_CMB',
        ... )
        >>> result = cmbOp(sky_map)
        >>> print(result)
    """

    factor: Float[Array, '...'] | float
    units: str = equinox.field(static=True)

    def __init__(
        self,
        frequencies: Float[Array, '...'],
        in_structure: PyTree[jax.ShapeDtypeStruct],
        units: str = 'K_CMB',
    ) -> None:
        self.units = units
        if units == 'K_CMB':
            self.factor = 1.0
        elif units == 'K_RJ':
            self.factor = K_RK_2_K_CMB(frequencies)

        super().__init__(frequencies, in_structure=in_structure)

    def sed(self) -> Float[Array, '...']:
        """
        Compute the spectral energy distribution for the CMB.

        Returns:
            Float[Array, '...']: The SED for the CMB.
        """
        return jnp.ones_like(self.frequencies) / jnp.expand_dims(self.factor, axis=-1)


class DustOperator(AbstractSEDOperator):
    """
    Operator for dust spectral energy distribution.

    Args:
        frequencies (Array): Array of frequencies.
        frequency0 (float, optional): Reference frequency. Defaults to 100.
        temperature (float | Array): Dust temperature.
        units (str, optional): Units for the operator ('K_CMB' or 'K_RJ'). Defaults to 'K_CMB'.
        temperature_patch_indices (Array | None, optional): Indices for patch-based temperature.
        beta (float | Array): Spectral index beta.
        beta_patch_indices (Array | None, optional): Indices for patch-based beta.
        in_structure (PyTree): Input structure.

    Attributes:
        temperature (Array): Dust temperature.
        beta (Array): Spectral index beta.
        factor (Array | float): Conversion factor based on the unit type.

    Example:
        >>> from furax.operators.seds import SynchrotronOperator
        >>> import jax.numpy as jnp
        >>> nu = jnp.array([30, 40, 100])  # Frequencies in GHz
        >>> in_structure = ...  # Define input structure (e.g., using HealpixLandscape)
        >>> beta_dust = 1.54  # Spectral index
        >>> temperature = 20.0  # Dust temperature
        >>> sky_map = ...  # Define sky map
        >>> dustOperator = SynchrotronOperator(
        ...     frequencies=nu,
        ...     frequency0=20.0,
        ...     beta=beta_dust,
        ...     temperature=temperature
        ...     in_structure=in_structure,
        ...     units='K_CMB',
        ... )
        >>> result = dustOperator(sky_map)
        >>> print(result)
    """

    temperature: Float[Array, '...']
    temperature_patch_indices: Int[Array, '...'] | None
    beta: Float[Array, '...']
    beta_patch_indices: Int[Array, '...'] | None
    factor: Float[Array, '...'] | float
    units: str = equinox.field(static=True)

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
        self.temperature = jnp.asarray(temperature)
        self.temperature_patch_indices = temperature_patch_indices
        self.beta = jnp.asarray(beta)
        self.beta_patch_indices = beta_patch_indices
        self.units = units

        if units == 'K_CMB':
            self.factor = K_RK_2_K_CMB(frequencies) / K_RK_2_K_CMB(frequency0)
        elif units == 'K_RJ':
            self.factor = 1.0

        super().__init__(
            frequencies,
            frequency0=frequency0,
            in_structure=in_structure,
        )

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
    """Spectral Energy Distribution (SED) operator for synchrotron emission.

    This operator models synchrotron emission based on a power-law SED
    with optional running of the spectral index.

    Attributes:
        beta_pl (Float[Array, '...']): Power-law spectral index values.
        beta_pl_patch_indices (Int[Array, '...'] | None):
             Optional indices for patch-specific beta values.
        nu_pivot (float): Pivot frequency in GHz for the running spectral index. Default is 1.0 GHz.
        running (float): Running of the spectral index. Default is 0.0.
        units (str): Output unit for the operator, either 'K_CMB' or 'K_RJ'.
        factor (Float[Array, '...'] | float): Conversion factor between units.

    Args:
        frequencies (Float[Array, '...']): Frequencies in GHz.
        frequency0 (float): Reference frequency for the SED. Default is 100 GHz.
        nu_pivot (float): Pivot frequency for running spectral index. Default is 1.0 GHz.
        running (float): Running of the spectral index. Default is 0.0.
        units (str): Units of the output, either 'K_CMB' or 'K_RJ'. Default is 'K_CMB'.
        beta_pl (float | Float[Array, '...']):
            Power-law spectral index or an array of indices.
        beta_pl_patch_indices (Int[Array, '...'] | None):
            Optional indices for patch-specific beta values.
        in_structure (PyTree[jax.ShapeDtypeStruct]): Input structure defining the shape of the data.

    Example:
        >>> from furax.operators.seds import SynchrotronOperator
        >>> import jax.numpy as jnp
        >>> nu = jnp.array([30, 40, 100])  # Frequencies in GHz
        >>> in_structure = ...  # Define input structure (e.g., using HealpixLandscape)
        >>> sky_map = ...  # Define sky map
        >>> beta_pl = -3.0  # Spectral index
        >>> synchrotron_operator = SynchrotronOperator(
        ...     frequencies=nu,
        ...     frequency0=20.0,
        ...     beta_pl=beta_pl,
        ...     in_structure=in_structure,
        ...     units='K_CMB',
        ... )
        >>> result = synchrotron_operator(sky_map)
        >>> print(result)
    """

    beta_pl: Float[Array, '...']
    beta_pl_patch_indices: Int[Array, '...'] | None
    nu_pivot: float = equinox.field(static=True)
    running: float = equinox.field(static=True)
    units: str = equinox.field(static=True)
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
        self.beta_pl = jnp.asarray(beta_pl)
        self.beta_pl_patch_indices = beta_pl_patch_indices
        self.nu_pivot = nu_pivot
        self.running = running
        self.units = units

        if units == 'K_CMB':
            self.factor = K_RK_2_K_CMB(frequencies) / K_RK_2_K_CMB(frequency0)
        elif units == 'K_RJ':
            self.factor = 1

        super().__init__(
            frequencies,
            frequency0=frequency0,
            in_structure=in_structure,
        )

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
        >>> from furax.operators.seds import CMBOperator, DustOperator,\
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
