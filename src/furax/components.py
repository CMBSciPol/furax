import dataclasses as dc
import sys
from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from typing import ClassVar, Literal, NamedTuple, Union, cast, get_args, overload

import jax
import jax.numpy as jnp

# Define a dataclass to hold parameters
import jax_dataclasses as jdc
import numpy as np
from astropy.cosmology import Planck15
from jaxtyping import Array, ArrayLike, Float, Integer, PyTree, ScalarLike
from scipy import constants

KeyArrayLike = ArrayLike

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

NumberType = Union[
    jnp.float32, jnp.int32, jnp.int16
]  # to be completed with all jax scalar number types
ScalarType = Union[jnp.bool_, NumberType]
DTypeLike = Union[
    str,  # like 'float32', 'int32'
    type[
        Union[
            bool, int, float, complex, ScalarType, np.bool_, np.number  # type: ignore[type-arg]  # noqa: E501
        ]
    ],
    np.dtype,  # type: ignore[type-arg]
]

ValidComponentsType = Literal['Synchrotron', 'Dust', 'CMB']

H_OVER_K = constants.h * 1e9 / constants.k
TCMB = Planck15.Tcmb(0).value


def K_RK_2_K_CMB(nu: ArrayLike) -> Array:
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
    return jnp.expm1(H_OVER_K * nu / TCMB) ** 2 / (
        jnp.exp(H_OVER_K * nu / TCMB) * (H_OVER_K * nu / TCMB) ** 2
    )


@jdc.pytree_dataclass
class CompParamsPytree(ABC):
    """Abstract base class for component parameters in cosmology.

    This class defines the structure and methods required for different components
    like Synchrotron, Dust, and CMB. It includes utilities to initialize, access,
    and modify parameter sets.
    See the subclasses `SynchParams`, `DustParam`, and `CMBParam` for specific implementations.

    Properties:
        size (int): Number of elements in the parameter arrays.
        dtype (DTypeLike): Data type of the parameters.
        structure (PyTree): JAX structure of the parameters.
        params (PyTree): Names of the parameters.
        n_params (int): Number of parameters for the component.

    Methods:
        zeros(): Initialize parameters with zeros.
        ones(): Initialize parameters with ones.
        full(): Initialize parameters with a specific fill value.
        normal(): Initialize parameters with normally distributed values.
        uniform(): Initialize parameters with uniformly distributed values.

    Example:
        >>> synch_params = SynchParams(beta_pl=jnp.array([1.5]))
        >>> print(synch_params.size)
        >>> print(synch_params.dtype)
    """

    component_type: ClassVar[ValidComponentsType]

    @property
    def size(self) -> int:
        """
        Number of elements in the parameter arrays.
        """
        return len(getattr(self, self.params[0]))

    @property
    def dtype(self) -> DTypeLike:
        """
        Data type of the parameters.
        """
        return cast(DTypeLike, getattr(self, self.params[0]).dtype)

    @property
    def structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        """
        structure of the pytree.
        """
        return self.structure_for(self.size, self.dtype, self.component_type)

    @property
    def params(self) -> PyTree:
        """
        Names of the parameters.
        """
        return self.params_for(self.component_type)

    @property
    def n_params(self) -> int:
        """
        Number of parameters for the component.
        """
        return self.n_params_for(self.component_type)

    @classmethod
    @overload
    def class_for(cls, component_type: Literal['Synchrotron']) -> type['SynchParams']: ...

    @classmethod
    @overload
    def class_for(cls, component_type: Literal['Dust']) -> type['DustParam']: ...

    @classmethod
    @overload
    def class_for(cls, component_type: Literal['CMB']) -> type['CMBParam']: ...

    @classmethod
    def class_for(cls, component_type: str) -> type['CompParamsPytreeType']:
        """
        return the class for the given component type.
        """
        if component_type not in get_args(ValidComponentsType):
            raise ValueError(f'Invalid Stokes parameters: {component_type!r}')
        requested_cls = {
            'Synchrotron': SynchParams,
            'Dust': DustParam,
            'CMB': CMBParam,
        }[component_type]
        return cast(type[CompParamsPytreeType], requested_cls)

    @classmethod
    def structure_for(
        cls, size: int, dtype: DTypeLike, component_type: ValidComponentsType
    ) -> PyTree[jax.ShapeDtypeStruct]:
        """
        Return the structure of the pytree for the given component type.
        """
        cls = CompParamsPytree.class_for(component_type)
        nb_params = cls.n_params_for(component_type)
        arrays = nb_params * [jax.ShapeDtypeStruct((size,), dtype)]
        return cls(*arrays)

    @classmethod
    def params_for(cls, component_type: ValidComponentsType) -> PyTree:
        cls = CompParamsPytree.class_for(component_type)
        members = dc.fields(cls)
        return [m.name for m in members]

    @classmethod
    def n_params_for(cls, component_type: ValidComponentsType) -> int:
        return len(cls.params_for(component_type))

    def __getitem__(self, index: Integer[Array, '...']) -> Self:
        arrays = [getattr(self, param)[index] for param in self.params]
        return type(self)(*arrays)

    @classmethod
    def zeros(cls, size: int = 1, dtype: DTypeLike | float = float) -> Self:
        """Create a parameter set initialized with zeros.

        Args:
            size (int): Number of elements in each parameter array.
            dtype (DTypeLike | float): Data type of the parameters.

        Returns:
            CompParamsPytree: An instance of the component parameters initialized with zeros.

        Example:
            >>> params = SynchParams.zeros(size=5)
            >>> print(params)
        """
        return cls.full(0, size, dtype)

    @classmethod
    def ones(cls, size: int = 1, dtype: DTypeLike | float = float) -> Self:
        """Create a parameter set initialized with ones.

        Args:
            size (int): Number of elements in each parameter array.
            dtype (DTypeLike | float): Data type of the parameters.

        Returns:
            CompParamsPytree: An instance of the component parameters initialized with ones.

        Example:
            >>> params = SynchParams.ones(size=3)
            >>> print(params)
        """
        return cls.full(1, size, dtype)

    @classmethod
    def full(cls, fill_value: ScalarLike, size: int = 1, dtype: DTypeLike | float = float) -> Self:
        """Create a parameter set filled with a specific value.

        Args:
            fill_value (ScalarLike): The value to fill the arrays with.
            size (int): Number of elements in each parameter array.
            dtype (DTypeLike | float): Data type of the parameters.

        Returns:
            CompParamsPytree: An instance of the component parameters
            initialized with the given value.

        Example:
            >>> params = SynchParams.full(fill_value=5, size=4)
            >>> print(params)
        """
        nb_params = cls.n_params_for(cls.component_type)
        arrays = nb_params * [jnp.full((size,), fill_value, dtype)]
        return cls(*arrays)

    @classmethod
    def normal(cls, key: KeyArrayLike, size: int = 1, dtype: DTypeLike = float) -> Self:
        """Create a parameter set initialized with normally distributed random values.

        Args:
            key (KeyArrayLike): PRNG key for random number generation.
            size (int): Number of elements in each parameter array.
            dtype (DTypeLike): Data type of the parameters.

        Returns:
            CompParamsPytree: An instance of the component parameters with random values.

        Example:
            >>> key = jax.random.PRNGKey(0)
            >>> params = SynchParams.normal(key, size=5)
            >>> print(params)
        """
        nb_params = cls.n_params_for(cls.component_type)
        keys = jax.random.split(key, nb_params)
        arrays = [jax.random.normal(key, (size,), dtype=dtype) for key in keys]
        return cls(*arrays)

    @classmethod
    def uniform(
        cls,
        key: KeyArrayLike,
        size: int = 1,
        dtype: DTypeLike = float,
        minval: float = 0.0,
        maxval: float = 1.0,
    ) -> Self:
        """Create a parameter set initialized with uniformly distributed random values.

        Args:
            key (KeyArrayLike): PRNG key for random number generation.
            size (int): Number of elements in each parameter array.
            dtype (DTypeLike): Data type of the parameters.
            minval (float): Minimum value of the uniform distribution.
            maxval (float): Maximum value of the uniform distribution.

        Returns:
            CompParamsPytree: An instance of the component parameters with random uniform values.

        Example:
            >>> key = jax.random.PRNGKey(0)
            >>> params = SynchParams.uniform(key, size=3)
            >>> print(params)
        """
        nb_params = cls.n_params_for(cls.component_type)
        keys = jax.random.split(key, nb_params)
        arrays = [
            jax.random.uniform(key, (size,), dtype=dtype, minval=minval, maxval=maxval)
            for key in keys
        ]
        return cls(*arrays)


@jdc.pytree_dataclass
class SynchParams(CompParamsPytree):
    """Class for synchrotron component parameters."""

    component_type: ClassVar[ValidComponentsType] = 'Synchrotron'
    beta_pl: Float[Array, '...']


@jdc.pytree_dataclass
class DustParam(CompParamsPytree):
    """Class for dust component parameters."""

    component_type: ClassVar[ValidComponentsType] = 'Dust'
    beta_d: Float[Array, '...']
    temp_d: Float[Array, '...']


@jdc.pytree_dataclass
class CMBParam(CompParamsPytree):
    """Class for CMB component parameters."""

    component_type: ClassVar[ValidComponentsType] = 'CMB'


CompParamsPytreeType = SynchParams | DustParam | CMBParam


class BatchingMode(Enum):
    NO_BATCH = (0,)
    BATCH_ALL = (1,)


# @jtu.register_static
class Component(ABC):

    def __init__(self):
        self.batching_mode = BatchingMode.NO_BATCH
        self.__impl = None

    @abstractmethod
    def evaluate(self, nu: ArrayLike, params: CompParamsPytree) -> Array: ...

    def set_batching_mode(self, mode: BatchingMode):
        self.batching_mode = mode
        self.__impl: Callable[[ArrayLike, CompParamsPytree], Array] | None = None

    def __call__(self, nu: ArrayLike, params: CompParamsPytreeType) -> Array:
        if self.__impl is not None:
            return self.__impl(nu, params)

        if self.batching_mode == BatchingMode.NO_BATCH:
            self.__impl = jax.vmap(self.evaluate, in_axes=(0, None))
        elif self.batching_mode == BatchingMode.BATCH_ALL:
            self.__impl = jax.vmap(self.evaluate, in_axes=(0, 0))

        return self.__impl(nu, params)


class CMB(Component):
    """CMB component model with constant SED.

    Evaluates a constant spectral energy distribution in `K_CMB` units or converts to `K_RJ`.

    Example:
        >>> cmb = CMB(units='K_CMB')
        >>> params = CMBParam()
        >>> nu = jnp.array([30, 40, 100])
        >>> sed = cmb(nu, params)
    """

    def __init__(self, units='K_CMB') -> None:
        self.units = units

        if units not in ['K_CMB', 'K_RJ']:
            raise ValueError('Unsupported units: %s' % units)
        super().__init__()

    def evaluate(self, nu: ArrayLike, params: CMBParam) -> Array:
        del params

        sed = jnp.ones_like(nu)

        if self.units == 'K_RJ':
            return sed / K_RK_2_K_CMB(nu)

        return sed[..., None]


class Synchrotron(Component):
    """Synchrotron component model with power-law SED.

    The spectral energy distribution (SED) follows a power law with optional running.

    .. math::
        S(\nu) = \\left( \frac{\nu}{\nu_0} \right)^{\beta_{\text{pl}}
        + r \\log\\left(\frac{\nu}{\nu_{\text{pivot}}}\right)}

    Example:
        >>> synch = Synchrotron(nu0=30)
        >>> params = SynchParams(beta_pl=jnp.array([1.5]))
        >>> nu = jnp.array([30, 40, 50])
        >>> sed = synch(nu, params)
    """

    def __init__(self, nu0, nu_pivot=1, running=0, units='K_CMB'):
        self.nu0 = nu0
        self.nu_pivot = nu_pivot
        self.running = running
        self.units = units
        super().__init__()

    def evaluate(self, nu: ArrayLike, params: SynchParams) -> Array:
        sed = (nu / self.nu0) ** (params.beta_pl + self.running * jnp.log(nu / self.nu_pivot))
        if self.units == 'K_CMB':
            sed *= K_RK_2_K_CMB(nu) / K_RK_2_K_CMB(self.nu0)
        return sed


class Dust(Component):
    """Dust component model with modified blackbody SED.

    The spectral energy distribution (SED) for dust is modeled as a modified blackbody.

    .. math::
        S(\nu) = \\left( \frac{e^{\frac{h \nu_0}{k T_d}} - 1}{e^{\frac{h \nu}{k T_d}} - 1} \right)
        \\left( \frac{\nu}{\nu_0} \right)^{1 + \beta_d}

    Example:
        >>> dust = Dust(nu0=353)
        >>> params = DustParam(beta_d=jnp.array([1.5]), temp_d=jnp.array([20.0]))
        >>> nu = jnp.array([100, 353, 545])
        >>> sed = dust(nu, params)
    """

    def __init__(self, nu0, units='K_CMB'):
        self.nu0 = nu0
        self.units = units
        super().__init__()

    def evaluate(self, nu: ArrayLike, params: DustParam) -> Array:
        sed = (
            (jnp.exp(self.nu0 / params.temp_d * H_OVER_K) - 1)
            / (jnp.exp(nu / params.temp_d * H_OVER_K) - 1)
            * (nu / self.nu0) ** (1 + params.beta_d)
        )
        if self.units == 'K_RJ':
            sed *= K_RK_2_K_CMB(nu) / K_RK_2_K_CMB(self.nu0)
        return sed


def SpectralParameters(*params: CompParamsPytree) -> NamedTuple:
    """
    Create a mixing matrix for multiple components based on their parameters.
    This is very usefull to as as a single argument in a likelihood function for example.
    This way, the gradient of the likelihood will be done on all the parameters at once.
    And the structure of the gradient will be the same as the MixingMatrix itself.

    Args:
        *params: Component parameter sets to form the mixing matrix.
         see CompParamsPytree for more details.

    Returns:
        NamedTuple: A named tuple representing the mixing matrix for each component.

    Example:
        >>> synch_params = SynchParams(beta_pl=jnp.array([1.5]))
        >>> dust_params = DustParam(beta_d=jnp.array([1.7]), temp_d=jnp.array([20.0]))
        >>> mixing_matrix = MixingMatrix(synch_params, dust_params)
    """
    assert all(
        isinstance(param, CompParamsPytree) for param in params
    ), 'All parameters should be instances of CompParamsPytree class'
    mixing_matrix_named_tuple = namedtuple(
        'MixingMatrix', [param.component_type for param in params]
    )
    return mixing_matrix_named_tuple(*params)


class MixingMatrixOperator:
    """Operator to evaluate a combined SED for multiple components.

    Combines the SEDs of different components into a single output
    by horizontally stacking their SED evaluations.
    The output of this operator is meant to be used as input
    in a DenseBlockDiagonalOperator.

    Example:
        >>> cmb = CMB(units='K_CMB')
        >>> synch = Synchrotron(nu0=30)
        >>> dust = Dust(nu0=30)
        >>> operator = MixingMatrixOperator(cmb, synch, dust)
        >>> params = MixingMatrix(CMBParam(), SynchParams(beta_pl=jnp.array([1.5])),
            ... DustParam(beta_d=jnp.array([1.5]), temp_d=jnp.array([20])))
        >>> nu = jnp.array([30, 100, 353])
        >>> result = operator(nu, params)
    """

    def __init__(self, *components: Component) -> None:
        assert all(
            isinstance(component, Component) for component in components
        ), 'All components should be instances of Component class'
        self.evaluators = [*components]

    def __call__(self, nu: ArrayLike, params: NamedTuple) -> Array:
        """Evaluate the combined SED for all components.

        Args:
            nu (ArrayLike): Frequencies for evaluation.
            params (NamedTuple): Parameters for each component.

        Returns:
            Array: The horizontally stacked SED evaluations for each component.
        """
        return jnp.hstack(
            [component(nu, param) for component, param in zip(self.evaluators, params)]
        )
