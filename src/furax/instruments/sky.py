import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pysm3
import pysm3.units as u
from jaxtyping import ArrayLike, DTypeLike, PyTree

from furax.landscapes import FrequencyLandscape, StokesPyTree, ValidStokesType

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@jax.tree_util.register_pytree_node_class
class FGBusterInstrumentPytree:
    """
    A PyTree-compatible class for representing an instrument as implemented in FGBuster framework.

    This class handles frequency, depth for intensity (depth_i),
    and depth for polarization (depth_p) for a given instrument.
    It provides utility methods for constructing instruments
    from various parameters, converting depths
    to desired units, and supporting JAX's PyTree structure.

    Attributes:
        frequency (ArrayLike): Array of frequency values.
        depth_i (ArrayLike): Depth values for intensity.
        depth_p (ArrayLike): Depth values for polarization.

    Methods:
      default_instrument(): Returns a default instrument with predefined parameters.
      from_depth_i(): Creates an instrument using
                      intensity depth and derives polarization depth.
      from_depth_p(): Creates an instrument using
                      polarization depth and derives intensity depth.
      from_params(): Directly creates an instrument from
                      given frequency, intensity depth, and polarization depth.
          depth_conversion(): Converts depths to a specified unit and dtype.
    Note:
        The class must be constructed with numpy arrays (not JAX arrays)
        in order to be pysm3 and astropy compatible.
        when depth_conversion() is called, the arrays are converted
        to JAX arrays and implicitly copied to GPU.

    Examples:
        >>> # Create a default instrument
        >>> instrument = FGBusterInstrumentPytree.default_instrument()

        >>> # Create an instrument from intensity depth
        >>> frequency = np.arange(10.0, 300, 30.0)
        >>> depth_i = (np.linspace(20, 40, 10) - 30) ** 2
        >>> instrument = FGBusterInstrumentPytree.from_depth_i(frequency, depth_i)

        >>> # Convert depth to micro-Kelvin CMB units
        >>> instrument.depth_conversion(unit='uK_CMB')
    """

    def __init__(self, frequency: ArrayLike, depth_i: ArrayLike, depth_p: ArrayLike):
        self.frequency: ArrayLike = frequency
        self.depth_i: ArrayLike = depth_i
        self.depth_p: ArrayLike = depth_p

    def tree_flatten(self) -> tuple:
        """
        Flattens the PyTree structure of the instrument.
        """
        return (self.frequency, self.depth_i, self.depth_p), None

    @classmethod
    def tree_unflatten(cls, _: Any, children: PyTree) -> 'FGBusterInstrumentPytree':
        """
        Reconstructs the PyTree structure of the instrument.
        """
        return cls(*children)

    @classmethod
    def default_instrument(cls) -> Self:
        """
        Returns a default instrument with predefined parameters.
        """

        frequency = np.arange(10.0, 300, 30.0)
        depth_p = (np.linspace(20, 40, 10) - 30) ** 2
        depth_i = (np.linspace(20, 40, 10) - 30) ** 2
        return cls(frequency, depth_i, depth_p)

    @classmethod
    def from_depth_i(cls, frequency: ArrayLike, depth_i: ArrayLike) -> 'FGBusterInstrumentPytree':
        """
        Creates an instrument using intensity depth and derives polarization depth.
        """
        depth_p = depth_i * np.sqrt(2)
        return cls(frequency, depth_i, depth_p)

    @classmethod
    def from_depth_p(cls, frequency: ArrayLike, depth_p: ArrayLike) -> 'FGBusterInstrumentPytree':
        """
        Creates an instrument using polarization depth and derives intensity depth.
        """
        depth_i = depth_p / jnp.sqrt(2)
        return cls(frequency, depth_i, depth_p)

    @classmethod
    def from_params(
        cls, frequency: ArrayLike, depth_i: ArrayLike, depth_p: ArrayLike
    ) -> 'FGBusterInstrumentPytree':
        """
        Directly creates an instrument from given
        frequency, intensity depth, and polarization depth.
        """
        return cls(frequency, depth_i, depth_p)

    def depth_conversion(self, unit: str = 'uK_CMB', dtype: DTypeLike = jnp.float32) -> None:
        """
        Converts depths to a specified unit and dtype.
        """
        self.depth_i *= u.arcmin * u.uK_CMB
        self.depth_p *= u.arcmin * u.uK_CMB

        self.depth_i = self.depth_i.to(
            getattr(u, unit) * u.arcmin,
            equivalencies=u.cmb_equivalencies(self.frequency * u.GHz),
        )
        self.depth_p = self.depth_p.to(
            getattr(u, unit) * u.arcmin,
            equivalencies=u.cmb_equivalencies(self.frequency * u.GHz),
        )

        # add axis for broadcasting
        self.depth_i = self.depth_i[:, None]
        self.depth_p = self.depth_p[:, None]

        self.depth_i = jnp.array(self.depth_i, dtype=dtype)
        self.depth_p = jnp.array(self.depth_p, dtype=dtype)
        self.frequency = jnp.array(self.frequency, dtype=dtype)


def get_sky(nside: int, tag: str = 'c1d0s0') -> pysm3.Sky:
    """
    Retrieves a sky model based on a specified resolution and component tag.

    This function creates a `pysm3.Sky` object with a given HEALPix resolution (`nside`)
    and a preset tag string (`tag`).
    The tag string defines which components (e.g., CMB, dust, synchrotron)
    will be included in the sky model.
    The tag is split into 2-character substrings representing each component.

    Args:
        nside (int): The nside parameter for the HEALPix resolution. Must be a power of 2.
        tag (str, optional): A preset tag defining the components of the sky model.
                             The default value 'c1d0s0' includes CMB, dust, and synchrotron.

    Returns:
        pysm3.Sky: A PySM3 Sky object containing the components specified by the tag.

    Examples:
        >>> # Get a sky model with default components (C1, D0 and S0) at nside 128
        >>> sky = get_sky(128)

        >>> # Get a sky model with custom components
        >>> sky = get_sky(64, tag='c1d1s1')  # Includes model C1, D1, and S1
    """
    preset_strings = [tag[i : i + 2] for i in range(0, len(tag), 2)]
    return pysm3.Sky(nside, preset_strings=preset_strings)


def get_observation(
    instrument: FGBusterInstrumentPytree,
    nside: int,
    tag='c1d0s0',
    add_noise: bool = False,
    stokes_type: ValidStokesType = 'IQU',
    dtype: DTypeLike = np.float32,
    unit: str = 'uK_CMB',
) -> StokesPyTree:
    """
    Generates a simulated sky observation using a given instrument and Sky model.

    This function combines emission from a sky model and a Gaussian random sky to simulate
    observations at the frequencies defined by the input instrument. Optionally, noise can
    be added to the observation based on the instrument's sensitivity (depth).
    The noise is added by multiplying the depth of
    the instrument and dividing by the resolution of the healpix map.
    The resolution is calculated using the formula
    .. math::

        sqrt(4 pi / (12 * nside^2)) * 180 / pi * 60.

    Args:
        instrument (FGBusterInstrumentPytree): The instrument to use for the observation,
                   including frequency and depth information.
        nside (int): The nside parameter for the HEALPix resolution.
        tag (str, optional): A preset string defining
            the components of the sky model. Defaults to 'c1d0s0'.
        add_noise (bool, optional): Whether to add noise to
            the observation. Defaults to False.
        stokes_type (ValidStokesType, optional): The Stokes components
            to include ('I', 'QU', 'IQU'). Defaults to 'IQU'.
        dtype (DTypeLike, optional): The data type for the output.
            Defaults to np.float32.
        unit (str, optional): The unit for the depth conversion.
            Defaults to 'uK_CMB'.

    Returns:
        StokesPyTree: The simulated sky observation, including Gaussian and emission components.

    Examples:
        >>> instrument = FGBusterInstrumentPytree.default_instrument()
        >>> nside = 128
        >>> stokes_type = 'IQU'
        >>> observation = get_observation(instrument,
              ... nside, stokes_type=stokes_type, add_noise=True)

        >>> # Generating an observation without noise
        >>> observation = get_observation(instrument,
              ... nside, stokes_type=stokes_type , add_noise=False)
    """
    pysm_sky = get_sky(nside, tag)
    landscapes = FrequencyLandscape(nside, instrument.frequency, stokes_type, dtype=dtype)
    gauss_sky = landscapes.zeros()

    if add_noise:
        instrument.depth_conversion(unit)
        resolution = jnp.sqrt(4 * jnp.pi / (12 * nside**2)) * 180 / jnp.pi * 60
        stoke_list = []
        match stokes_type:
            case 'I':
                stoke_list = [gauss_sky.i * (instrument.depth_i / resolution)]
            case 'QU':
                stoke_list = [
                    gauss_sky.q * (instrument.depth_p / resolution),
                    gauss_sky.u * (instrument.depth_p / resolution),
                ]
            case 'IQU':
                stoke_list = [
                    gauss_sky.i * (instrument.depth_i / resolution),
                    gauss_sky.q * (instrument.depth_p / resolution),
                    gauss_sky.u * (instrument.depth_p / resolution),
                ]
            case _:
                raise ValueError(f'Invalid Stokes type {stokes_type}')

        gauss_sky = StokesPyTree.from_stokes(*stoke_list)

    stoke_arrays: ArrayLike = np.array([np.zeros(gauss_sky.shape) for _ in stokes_type]).transpose(
        1, 0, 2
    )
    match stokes_type:
        case 'I':
            # From a two dimension array for example (3 , 48) take a[0].reshape(1 , -1)
            stoke_slice = slice(0, 1)
        case 'QU':
            stoke_slice = slice(1, 3)
        case 'IQU':
            stoke_slice = slice(None)
        case _:
            raise ValueError(f'Invalid Stokes type {stokes_type}')

    for freq_indx, freq in enumerate(instrument.frequency):
        emission = pysm_sky.get_emission(freq * u.GHz)

        emission = emission.to(getattr(u, unit), equivalencies=u.cmb_equivalencies(freq * u.GHz))
        emission = emission.value[stoke_slice]

        stoke_arrays[freq_indx] = emission

    # From numpy array to Py²Tree
    stokes_array = stoke_arrays.transpose(1, 0, 2)
    stokes_array = jnp.array(stokes_array, dtype=dtype)
    stokes_pytree = StokesPyTree.from_stokes(*stokes_array)
    # Add emission to gaussian Sky
    emission_sky_pytree = jax.tree.map(lambda x, y: x + y, gauss_sky, stokes_pytree)

    return emission_sky_pytree