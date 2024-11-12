from abc import abstractmethod

import equinox
import jax
from jax._src.api import F
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
from scipy import constants
from astropy.cosmology import Planck15, units

from furax._base.diagonal import BroadcastDiagonalOperator

H_OVER_K = constants.h * 1e9 / constants.k
TCMB = Planck15.Tcmb(0).value  # type: ignore


def K_RK_2_K_CMB(nu: Array | Float) -> Array:
    """Convert Rayleigh-Jeans brightness temperature to CMB temperature.

    .. math::
        T_{CMB} = \frac{(e^{\frac{h \nu}{k T_{CMB}}} - 1)^2}{(e^{\frac{h \nu}{k T_{CMB}}})
        \\left( \frac{h \nu}{k T_{CMB}} \right)^2}

    Args:
        nu (ArrayLike): Frequency in GHz.

    Returns:
        ArrayLike: Conversion factor from Rayleigh-Jeans to CMB temperature.

    Example:
        >>> nu = jnp.array([30, 40, 100])
        >>> conversion = K_RK_2_K_CMB(nu)
        >>> print(conversion)
    """
    return jnp.expm1(
        H_OVER_K * nu / TCMB)**2 / (jnp.exp(H_OVER_K * nu / TCMB) *
                                    (H_OVER_K * nu / TCMB)**2)



class AbstractSEDOperator(BroadcastDiagonalOperator):
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
        self.frequencies = frequencies.reshape((len(frequencies), ) +
                                               tuple(1 for _ in input_shape))
        self.frequency0 = frequency0
        super().__init__(self.sed(), in_structure=in_structure)

    @staticmethod
    def _get_input_shape(
            in_structure: PyTree[jax.ShapeDtypeStruct]) -> tuple[int, ...]:
        input_shapes = set(leaf.shape
                           for leaf in jax.tree.leaves(in_structure))
        if len(input_shapes) != 1:
            raise ValueError(
                f'the leaves of the input do not have the same shape: {in_structure}'
            )
        return input_shapes.pop()

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    @abstractmethod
    def sed(self) -> Float[Array, '...']:
        ...

    @staticmethod
    def _get_at(values: Float[Array, '...'],
                indices: Int[Array, '...'] | None) -> Float[Array, '...']:
        if indices is None:
            return values
        return values[..., indices]


class CMBOperator(AbstractSEDOperator):
    factor: Float[Array, '...'] | float
    units: str = equinox.field(static=True)

    def __init__(self,
                 frequencies: Float[Array, '...'],
                 in_structure: PyTree[jax.ShapeDtypeStruct],
                 units: str = 'K_CMB') -> None:

        self.units = units
        if units == 'K_CMB':
            self.factor = 1.0
        elif units == 'K_RJ':
            self.factor = K_RK_2_K_CMB(frequencies)

        super().__init__(frequencies, in_structure=in_structure)

    def sed(self) -> Float[Array, '...']:
        return jnp.ones_like(self.frequencies) / jnp.expand_dims(self.factor,
                                                                 axis=-1)


class DustOperator(AbstractSEDOperator):
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
            jnp.expm1(self.frequency0 / self.temperature * H_OVER_K) /
            jnp.expm1(self.frequencies / self.temperature * H_OVER_K),
            self.temperature_patch_indices,
        )
        b = self._get_at((self.frequencies / self.frequency0)**(1 + self.beta),
                         self.beta_patch_indices)
        sed = (t*b)  * jnp.expand_dims(self.factor , axis=-1)
        return sed


class SynchrotronOperator(AbstractSEDOperator):
    beta_pl: Float[Array, '...']
    beta_pl_patch_indices: Int[Array, '...'] | None
    nu_pivot: float = equinox.field(static=True)
    running: float = equinox.field(static=True)
    units: str = equinox.field(static=True)
    factor: Float[Array, '...'] | float
    units: str = equinox.field(static=True)

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
            ((self.frequencies / self.frequency0)**(self.beta_pl +
             self.running * jnp.log(self.frequencies / self.nu_pivot))),
            self.beta_pl_patch_indices,
        )

        sed = self._get_at((self.frequencies / self.frequency0)**self.beta_pl, self.beta_pl_patch_indices)
        sed *= jnp.expand_dims(self.factor, axis=-1)    

        return sed
