from abc import abstractmethod

import equinox
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
from scipy import constants

from furax._base.diagonal import BroadcastDiagonalOperator

H_OVER_K = constants.h / constants.k


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

    def __init__(self, frequencies: Float[Array, '...'],
                 in_structure: PyTree[jax.ShapeDtypeStruct]) -> None:
        super().__init__(frequencies, in_structure=in_structure)

    def sed(self) -> Float[Array, '...']:
        return jnp.ones_like(self.frequencies)


class DustOperator(AbstractSEDOperator):
    temperature: Float[Array, '...']
    temperature_patch_indices: Int[Array, '...'] | None
    beta: Float[Array, '...']
    beta_patch_indices: Int[Array, '...'] | None

    def __init__(
        self,
        frequencies: Float[Array, '...'],
        *,
        frequency0: float = 100e9,
        temperature: float | Float[Array, '...'],
        temperature_patch_indices: Int[Array, '...'] | None = None,
        beta: float | Float[Array, '...'],
        beta_patch_indices: Int[Array, '...'] | None = None,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:

        # if temperature_patch_indices is None:
        #     assert jnp.asarray(temperature).shape == (
        #     ) or jnp.asarray(temperature).shape == (1, )
        # else:
        #     assert jnp.asarray(temperature).shape == jnp.unique(
        #         temperature_patch_indices).shape

        # if beta_patch_indices is None:
        #     assert jnp.asarray(beta).shape == (
        #     ) or jnp.asarray(beta).shape == (1, )
        # else:
        #     assert jnp.asarray(beta).shape == jnp.unique(
        #         beta_patch_indices).shape

        self.temperature = jnp.asarray(temperature)
        self.temperature_patch_indices = temperature_patch_indices
        self.beta = jnp.asarray(beta)
        self.beta_patch_indices = beta_patch_indices
        super().__init__(
            frequencies,
            frequency0=frequency0,
            in_structure=in_structure,
        )

    def sed(self) -> Float[Array, '...']:
        t = self._get_at(
            jnp.expm1(self.frequency0 / self.temperature * H_OVER_K) /
            jnp.expm1(self.frequencies / self.temperature * H_OVER_K) *
            (self.frequencies / self.frequency0),
            self.temperature_patch_indices,
        )
        b = self._get_at(1 + self.beta, self.beta_patch_indices)

        sed = t**b
        return sed


class SynchrotronOperator(AbstractSEDOperator):
    beta_pl: Float[Array, '...']
    beta_pl_patch_indices: Int[Array, '...'] | None
    nu_pivot: float = equinox.field(static=True)
    running: float = equinox.field(static=True)
    units: str = equinox.field(static=True)

    def __init__(
        self,
        frequencies: Float[Array, '...'],
        *,
        frequency0: float = 100e9,
        nu_pivot: float = 1.0,
        running: float = 0.0,
        units: str = 'K_CMB',
        beta_pl: float | Float[Array, '...'],
        beta_pl_patch_indices: Int[Array, '...'] | None = None,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        # if beta_pl_patch_indices is None:
        #     assert jnp.asarray(beta_pl).shape == (
        #     ) or jnp.asarray(beta_pl).shape == (1, )
        # else:
        #     assert jnp.asarray(beta_pl).shape == jnp.unique(
        #         beta_pl_patch_indices).shape

        self.beta_pl = jnp.asarray(beta_pl)
        self.beta_pl_patch_indices = beta_pl_patch_indices
        self.nu_pivot = nu_pivot
        self.running = running
        self.units = units
        super().__init__(
            frequencies,
            frequency0=frequency0,
            in_structure=in_structure,
        )

    def sed(self) -> Float[Array, '...']:
        sed = self._get_at(
            ((self.frequencies / self.frequency0)**self.beta_pl +
             self.running + jnp.log(self.frequencies / self.frequency0)),
            self.beta_pl_patch_indices,
        )
        return sed


# class Synchrotron(Component):

#     def __init__(self, nu0, nu_pivot=1, running=0, units='K_CMB'):
#         self.nu0 = nu0
#         self.nu_pivot = nu_pivot
#         self.running = running
#         self.units = units
#         super().__init__()

#     def evaluate(self, nu: Array, params: SynchParams) -> Array:
#         sed = (nu / self.nu0)**(params.beta_pl +
#                                 self.running * jnp.log(nu / self.nu_pivot))
#         if self.units == 'K_CMB':
#             sed *= K_RK_2_K_CMB(nu) / K_RK_2_K_CMB(self.nu0)
#         return sed
