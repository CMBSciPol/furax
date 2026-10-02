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
from jaxtyping import Array, ArrayLike, Float, Inexact, Int, PyTree
from scipy import constants

from furax import AbstractLinearOperator, BlockRowOperator, BroadcastDiagonalOperator, diagonal

_H_OVER_K_GHZ = constants.h * 1e9 / constants.k
_T_CMB = Planck15.Tcmb(0).value

__all__ = [
    'AbstractSEDOperator',
    'CMBOperator',
    'DustOperator',
    'SynchrotronOperator',
    'mixing_matrix',
    'MixingMatrixOperator',
]


def k_rj_to_k_cmb(nu: ArrayLike) -> Array:
    r"""Conversion factor from Rayleigh-Jeans brightness temperature to CMB temperature.

    With $x = h \nu / k T_{CMB}$, the factor is

    $$
    \frac{(e^x - 1)^2}{e^x x^2}.
    $$

    Args:
        nu: Frequency in GHz.

    Returns:
        The factor that multiplies a temperature in $K_{RJ}$ to give it in $K_{CMB}$.

    Examples:
        >>> [round(float(f), 3) for f in k_rj_to_k_cmb(jnp.array([30.0, 100.0, 353.0]))]
        [1.023, 1.287, 12.905]
    """
    x = _H_OVER_K_GHZ * jnp.asarray(nu) / _T_CMB
    return jnp.expm1(x) ** 2 / (jnp.exp(x) * x**2)


def _check_units(units: str) -> None:
    if units not in ('K_CMB', 'K_RJ'):
        raise ValueError(f"Unknown units: {units}. Expected 'K_CMB' or 'K_RJ'.")


def _per_pixel(
    value: Float[Array, '...'], patch_indices: Int[Array, ' pix'] | None
) -> Float[Array, '...']:
    """Expand a per-patch spectral parameter to one value per pixel."""
    if patch_indices is None or value.ndim == 0:
        return value
    return value[patch_indices]


class AbstractSEDOperator(BroadcastDiagonalOperator):
    """Abstract base class for Spectral Energy Distribution (SED) operators.

    An SED operator scales the sky map of one astrophysical component to its emission in each
    frequency channel. The input is a map without frequency axis, e.g. a Stokes map of shape
    `(n_stokes, n_pix)`; the output inserts a frequency axis before the pixel axis, giving a map
    of shape `(n_stokes, n_freq, n_pix)`.

    Subclasses implement `sed`.

    Attributes:
        frequencies: Observation frequencies [GHz].
        units: Output units, `'K_CMB'` or `'K_RJ'`.
    """

    frequencies: Float[Array, ' freq']
    units: str = field(metadata={'static': True})

    def __init__(
        self,
        frequencies: Float[ArrayLike, ' freq'],
        *,
        units: str,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        _check_units(units)
        object.__setattr__(self, 'frequencies', jnp.asarray(frequencies, dtype=float))
        object.__setattr__(self, 'units', units)
        super().__init__(
            self.sed(), axis_destination=(-2, -1), insert_axes=-2, in_structure=in_structure
        )

    @abstractmethod
    def sed(self) -> Float[Array, 'freq pix'] | Float[Array, 'freq 1']:
        """Return the SED, sampled at the operator frequencies.

        Returns:
            The SED, with a trailing axis of length 1 when it does not depend on the pixel.
        """


class CMBOperator(AbstractSEDOperator):
    r"""Operator for the Cosmic Microwave Background (CMB) spectral energy distribution.

    The CMB has a blackbody spectrum at $T_{CMB} \approx 2.725$ K. In $K_{CMB}$ units, the SED is
    unity at all frequencies. In $K_{RJ}$ units, it is the inverse of `k_rj_to_k_cmb`.

    Attributes:
        frequencies: Observation frequencies [GHz].
        units: Output units, `'K_CMB'` or `'K_RJ'`.

    Examples:
        >>> from furax.obs.landscapes import HealpixLandscape
        >>> landscape = HealpixLandscape(nside=8, stokes='IQU')
        >>> cmb = CMBOperator(jnp.array([30.0, 40.0, 100.0]), in_structure=landscape.structure)
        >>> cmb(landscape.ones()).shape
        (3, 768)
    """

    def __init__(
        self,
        frequencies: Float[ArrayLike, ' freq'],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        units: str = 'K_CMB',
    ) -> None:
        super().__init__(frequencies, units=units, in_structure=in_structure)

    def sed(self) -> Float[Array, 'freq 1']:
        sed = jnp.ones_like(self.frequencies)
        if self.units == 'K_RJ':
            sed /= k_rj_to_k_cmb(self.frequencies)
        return sed[:, None]


class DustOperator(AbstractSEDOperator):
    r"""Operator for the thermal dust spectral energy distribution.

    Dust emission is modelled as a modified blackbody. In $K_{RJ}$ units, the SED is

    $$
    \left(\frac{\nu}{\nu_0}\right)^{1 + \beta} \frac{e^{h\nu_0/kT} - 1}{e^{h\nu/kT} - 1}.
    $$

    The temperature $T$ and spectral index $\beta$ may vary across the sky: given per-patch
    values, the patch indices assign a patch to each pixel.

    Attributes:
        frequencies: Observation frequencies [GHz].
        frequency0: Reference frequency $\nu_0$ [GHz].
        temperature: Dust temperature [K], scalar or one value per patch.
        temperature_patch_indices: Patch index of each pixel for `temperature`.
        beta: Spectral index, scalar or one value per patch.
        beta_patch_indices: Patch index of each pixel for `beta`.
        units: Output units, `'K_CMB'` or `'K_RJ'`.

    Examples:
        >>> from furax.obs.landscapes import HealpixLandscape
        >>> landscape = HealpixLandscape(nside=8, stokes='IQU')
        >>> dust = DustOperator(
        ...     jnp.array([100.0, 143.0, 217.0, 353.0]),
        ...     frequency0=353.0,
        ...     temperature=20.0,
        ...     beta=1.54,
        ...     in_structure=landscape.structure,
        ... )
        >>> dust(landscape.ones()).shape
        (4, 768)
    """

    frequency0: float = field(metadata={'static': True})
    temperature: Float[Array, '...']
    temperature_patch_indices: Int[Array, ' pix'] | None
    beta: Float[Array, '...']
    beta_patch_indices: Int[Array, ' pix'] | None

    def __init__(
        self,
        frequencies: Float[ArrayLike, ' freq'],
        *,
        frequency0: float = 100,
        temperature: float | Float[Array, ' patch'],
        units: str = 'K_CMB',
        temperature_patch_indices: Int[Array, ' pix'] | None = None,
        beta: float | Float[Array, ' patch'],
        beta_patch_indices: Int[Array, ' pix'] | None = None,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        object.__setattr__(self, 'frequency0', frequency0)
        object.__setattr__(self, 'temperature', jnp.asarray(temperature))
        object.__setattr__(self, 'temperature_patch_indices', temperature_patch_indices)
        object.__setattr__(self, 'beta', jnp.asarray(beta))
        object.__setattr__(self, 'beta_patch_indices', beta_patch_indices)
        super().__init__(frequencies, units=units, in_structure=in_structure)

    def sed(self) -> Float[Array, 'freq pix'] | Float[Array, 'freq 1']:
        nu = self.frequencies[:, None]
        temperature = _per_pixel(self.temperature, self.temperature_patch_indices)
        beta = _per_pixel(self.beta, self.beta_patch_indices)
        sed = (nu / self.frequency0) ** (1 + beta)
        sed *= jnp.expm1(_H_OVER_K_GHZ * self.frequency0 / temperature)
        sed /= jnp.expm1(_H_OVER_K_GHZ * nu / temperature)
        if self.units == 'K_CMB':
            sed *= k_rj_to_k_cmb(nu) / k_rj_to_k_cmb(self.frequency0)
        return jnp.broadcast_to(sed, (nu.shape[0], sed.shape[-1]))


class SynchrotronOperator(AbstractSEDOperator):
    r"""Operator for the synchrotron spectral energy distribution.

    Synchrotron emission is modelled as a power law with an optional running of the spectral
    index. In $K_{RJ}$ units, the SED is

    $$
    \left(\frac{\nu}{\nu_0}\right)^{\beta + r \log(\nu / \nu_{pivot})}.
    $$

    The spectral index $\beta$ may vary across the sky: given per-patch values, the patch indices
    assign a patch to each pixel.

    Attributes:
        frequencies: Observation frequencies [GHz].
        frequency0: Reference frequency $\nu_0$ [GHz].
        beta_pl: Power-law spectral index, scalar or one value per patch.
        beta_pl_patch_indices: Patch index of each pixel for `beta_pl`.
        nu_pivot: Pivot frequency $\nu_{pivot}$ of the running [GHz].
        running: Running $r$ of the spectral index.
        units: Output units, `'K_CMB'` or `'K_RJ'`.

    Examples:
        >>> from furax.obs.landscapes import HealpixLandscape
        >>> landscape = HealpixLandscape(nside=8, stokes='IQU')
        >>> synchrotron = SynchrotronOperator(
        ...     jnp.array([30.0, 44.0, 70.0]),
        ...     frequency0=30.0,
        ...     beta_pl=-3.0,
        ...     in_structure=landscape.structure,
        ... )
        >>> synchrotron(landscape.ones()).shape
        (3, 768)
    """

    frequency0: float = field(metadata={'static': True})
    beta_pl: Float[Array, '...']
    beta_pl_patch_indices: Int[Array, ' pix'] | None
    nu_pivot: float = field(metadata={'static': True})
    running: float = field(metadata={'static': True})

    def __init__(
        self,
        frequencies: Float[ArrayLike, ' freq'],
        *,
        frequency0: float = 100,
        nu_pivot: float = 1.0,
        running: float = 0.0,
        units: str = 'K_CMB',
        beta_pl: float | Float[Array, ' patch'],
        beta_pl_patch_indices: Int[Array, ' pix'] | None = None,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        object.__setattr__(self, 'frequency0', frequency0)
        object.__setattr__(self, 'beta_pl', jnp.asarray(beta_pl))
        object.__setattr__(self, 'beta_pl_patch_indices', beta_pl_patch_indices)
        object.__setattr__(self, 'nu_pivot', nu_pivot)
        object.__setattr__(self, 'running', running)
        super().__init__(frequencies, units=units, in_structure=in_structure)

    def sed(self) -> Float[Array, 'freq pix'] | Float[Array, 'freq 1']:
        nu = self.frequencies[:, None]
        beta = _per_pixel(self.beta_pl, self.beta_pl_patch_indices)
        sed = (nu / self.frequency0) ** (beta + self.running * jnp.log(nu / self.nu_pivot))
        if self.units == 'K_CMB':
            sed *= k_rj_to_k_cmb(nu) / k_rj_to_k_cmb(self.frequency0)
        return jnp.broadcast_to(sed, (nu.shape[0], sed.shape[-1]))


def mixing_matrix(**blocks: AbstractSEDOperator) -> AbstractLinearOperator:
    """Combine named SED operators into a mixing matrix.

    Args:
        **blocks: SED operators, keyed by component name.

    Returns:
        The operator mapping a dictionary of component maps, with the same keys as `blocks`, to the
        sum of their frequency maps.

    Examples:
        >>> from furax.obs.landscapes import HealpixLandscape
        >>> landscape = HealpixLandscape(nside=8, stokes='IQU')
        >>> nu = jnp.array([30.0, 40.0, 100.0])
        >>> A = mixing_matrix(
        ...     cmb=CMBOperator(nu, in_structure=landscape.structure),
        ...     dust=DustOperator(
        ...         nu, frequency0=150.0, temperature=20.0, beta=1.54,
        ...         in_structure=landscape.structure,
        ...     ),
        ... )
        >>> A({'cmb': landscape.ones(), 'dust': landscape.ones()}).shape
        (3, 768)
    """
    return BlockRowOperator(blocks).reduce()


@deprecated('Use mixing_matrix')
def MixingMatrixOperator(**blocks: AbstractSEDOperator) -> AbstractLinearOperator:
    """Deprecated alias of `mixing_matrix`."""
    return mixing_matrix(**blocks)


@deprecated('Should use a DiagonalOperator')
@diagonal
class NoiseDiagonalOperator(AbstractLinearOperator):
    """Constructs a diagonal noise operator.

    This operator applies a noise vector (in a PyTree structure) in an element‐wise
    multiplication to an input data PyTree.

    Attributes:
        vector: PyTree of arrays representing the noise values.
        in_structure: Input structure (PyTree[jax.ShapeDtypeStruct]) specifying the shape and dtype.

    Examples:
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
        StokesIQU(ShapeDtypeStruct(shape=(3, 10, 49152), dtype=float64))
    """

    vector: PyTree[Inexact[Array, '...']]

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return jax.tree.map(lambda v, leaf: v * leaf, self.vector, x)

    def inverse(self) -> AbstractLinearOperator:
        return NoiseDiagonalOperator(  # ty: ignore[deprecated]
            vector=1 / self.vector, in_structure=self.in_structure
        )

    def as_matrix(self) -> Any:
        return jax.tree.map(lambda x: jnp.diag(x.flatten()), self.vector)
