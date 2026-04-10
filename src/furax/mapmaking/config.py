from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import yaml
from apischema import deserialize, deserializer, serialize, serializer
from apischema.conversions import Conversion
from jax.typing import DTypeLike

from furax.obs.landscapes import ProjectionType
from furax.obs.stokes import ValidStokesType

# apischema serializes IntEnum by value (integer) by default; override to use the name instead
# so that YAML config files show e.g. 'CAR' rather than '0'.
serializer(Conversion(lambda p: p.name, source=ProjectionType, target=str))
deserializer(Conversion(lambda s: ProjectionType[s], source=str, target=ProjectionType))


class Methods(Enum):
    BINNED = 'Binned'
    MAXL = 'ML'
    TWOSTEP = 'TwoStep'
    ATOP = 'ATOP'


@dataclass
class SolverConfig:
    rtol: float = 1e-6
    atol: float = 0
    max_steps: int = 1_000


@dataclass
class NoiseFitConfig:
    nperseg: int = 2_048
    """Welch window length in samples for PSD estimation."""

    max_iter: int = 100
    """Maximum number of iterations"""

    tol: float = 1e-10
    """Relative minimiser tolerance (step size and function value change)"""

    min_freq_nyquist: float = 1e-8
    """Only use f >= min_freq * nyquist for noise fitting"""

    max_freq_nyquist: float = 1
    """Only use f < max_freq * nyquist for noise fitting"""

    low_freq_nyquist: float = 0.02
    """The PSD at f < low_freq * nyquist is assumed to be dominated by 1/f noise"""

    high_freq_nyquist: float = 0.02
    """The PSD at f > high_freq * nyquist is assumed to be dominated by white noise"""

    mask_hwp_harmonics: bool = True
    """Mask HWP harmonics: 1f, 2f, 4f"""

    mask_ptc_harmonics: bool = False
    """Mask PTC harmonics: 1f, 2f"""

    freq_mask_width: float = 0.5
    """Full width [Hz] of the frequency mask (if used) around HWP and PTC harmonics"""

    ptc_freq: float = 1.4
    """PTC frequency [Hz] used for masking (if used)"""


@dataclass
class NoiseConfig:
    """Configuration for noise modelling.

    ``fit_from_data`` controls where the noise model comes from:

    - ``True`` (default): fit a noise model directly from the TOD power spectral density.
    - ``False``: load precomputed noise parameters from the data pipeline.

    ``correlation_length`` sets the Toeplitz bandwidth (in samples) of the inverse-noise
    operator.  It is only used by the atmospheric (1/f) noise model and is ignored when
    ``white`` is True.
    """

    white: bool = True
    """Use a white (diagonal) noise model.  Set to False for the atmospheric (1/f) model."""

    fit_from_data: bool = True
    """Fit the noise model from the TOD PSD (True) or load it from the data pipeline (False)."""

    correlation_length: int = 1_000
    """Toeplitz bandwidth in samples.  Only relevant for the atmospheric (1/f) noise model."""

    fitting: NoiseFitConfig = field(default_factory=NoiseFitConfig)
    """Options controlling PSD estimation and model fitting."""


@dataclass
class HealpixConfig:
    """Configuration for a HEALPix output map.

    Example:
        In a YAML config file:

        healpix:
          nside: 512
    """

    nside: int = 512
    ordering: Literal['nest', 'ring'] = 'ring'

    def __post_init__(self) -> None:
        if self.ordering == 'nest':
            raise ValueError('NESTED ordering not supported')


@dataclass
class SkyPatch:
    """Explicit rectangular sky patch for WCS map construction.

    Example:
        In a YAML config file:

        patch:
          center: [30.0, -10.0]  # ra, dec in degrees
          width: 20.0
          height: 10.0
    """

    center: tuple[float, float]
    """Center ``(ra, dec)`` in degrees."""

    width: float
    """Width in degrees."""

    height: float
    """Height in degrees."""


@dataclass
class WCSConfig:
    """Configuration for a WCS-projected output map.

    ``projection`` applies to all modes except ``geometry_file``, where it is read from the file.

    The map extent is determined by exactly one of three mutually exclusive modes:

    1. **geometry_file**: read shape and WCS directly from a FITS/HDF file via
       ``pixell.enmap.read_map_geometry``. All other fields are ignored.
    2. **patch**: build a rectangular patch of sky at the given ``resolution``.
    3. **auto** (no geometry specified): scan the observations to compute each observation's
       bounding box, take their union, and pixelise at the given ``resolution``.

    Examples:
        In a YAML config file:

        # Auto footprint at 4 arcmin resolution
        car:
          resolution: 4.0

        # Explicit patch
        car:
          resolution: 4.0
          patch:
            center: [30.0, -10.0]
            width: 20.0
            height: 10.0

        # Geometry from file
        car:
          geometry_file: /path/to/map.fits
    """

    projection: ProjectionType = ProjectionType.CAR
    """WCS projection type."""

    resolution: float = 4.0
    """Pixel resolution in arcminutes."""

    geometry_file: str | None = None
    """Path to a FITS or HDF map file from which to read the output geometry."""

    patch: SkyPatch | None = None
    """Explicit sky patch definition. Mutually exclusive with ``geometry_file``."""

    def __post_init__(self) -> None:
        if self.geometry_file is not None and self.patch is not None:
            raise ValueError('geometry_file and patch are mutually exclusive.')

    @property
    def has_geometry(self) -> bool:
        return self.geometry_file is not None or self.patch is not None


@dataclass
class LandscapeConfig:
    stokes: ValidStokesType = 'IQU'
    healpix: HealpixConfig | None = None
    wcs: WCSConfig | None = None

    def __post_init__(self) -> None:
        if (self.healpix is None) == (self.wcs is None):
            raise ValueError('exactly one of healpix or wcs must be set.')


@dataclass
class _PolyTemplateConfig:
    max_poly_order: int = 3


@dataclass
class _ScanSynchronousTemplateConfig:
    min_poly_order: int = 3
    max_poly_order: int = 7


@dataclass
class _HWPSynchronousTemplateConfig:
    n_harmonics: int = 3


@dataclass
class _AzimuthHWPSynchronousTemplateConfig:
    n_polynomials: int = 4
    n_harmonics: int = 4
    split_scans: bool = False


@dataclass
class _BinAzimuthHWPSynchronousTemplateConfig:
    n_azimuth_bins: int = 4
    n_harmonics: int = 4
    interpolate_azimuth: bool = False
    smooth_interpolation: bool = False


@dataclass
class _GroundTemplateConfig:
    azimuth_resolution: float = 0.05  # ~3 deg
    elevation_resolution: float = 0.05  # ~3 deg


@dataclass
class TemplatesConfig:
    polynomial: _PolyTemplateConfig | None = None
    scan_synchronous: _ScanSynchronousTemplateConfig | None = None
    hwp_synchronous: _HWPSynchronousTemplateConfig | None = None
    azhwp_synchronous: _AzimuthHWPSynchronousTemplateConfig | None = None
    binazhwp_synchronous: _BinAzimuthHWPSynchronousTemplateConfig | None = None
    ground: _GroundTemplateConfig | None = None
    regularization: float = 0.0

    @classmethod
    def full_defaults(cls) -> 'TemplatesConfig':
        """Create a template config with default values for all templates."""
        return cls(
            polynomial=_PolyTemplateConfig(),
            scan_synchronous=_ScanSynchronousTemplateConfig(),
            hwp_synchronous=_HWPSynchronousTemplateConfig(),
            azhwp_synchronous=_AzimuthHWPSynchronousTemplateConfig(),
            binazhwp_synchronous=_BinAzimuthHWPSynchronousTemplateConfig(),
            ground=_GroundTemplateConfig(),
        )

    @property
    def empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))


@dataclass
class GapFillingConfig:
    """Specific gap-filling options"""

    seed: int = 286502183
    """An integer seed for the noise realization"""

    max_steps: int = 50
    """The maximum number of iteration steps to invert the system"""

    rtol: float = 1e-4
    """The relative tolerance of the solver for the gap-filling solve"""


@dataclass
class GapsConfig:
    """Configuration options related to the treatment of gaps"""

    fill: bool = True
    """Fill data gaps with synthetic noise-like samples"""

    fill_options: GapFillingConfig = field(default_factory=GapFillingConfig)
    """Options to pass to the gap-filling operator"""

    nested_pcg: bool = False
    """Use the nested PCG method for gap treatment"""


@dataclass
class PointingConfig:
    """Configuration options for pointing computation.

    ``interpolation`` controls how the sky map is sampled:

    - ``'nearest'``: nearest-neighbor (default, fastest).
    - ``'bilinear'``: bilinear interpolation using the four nearest pixels.
    """

    on_the_fly: bool = True
    """Compute pointing on the fly instead of pre-computing pixel indices."""

    chunk_size: int = 32
    """Number of detector chunks to process at a time when computing pointing on the fly."""

    interpolation: Literal['nearest', 'bilinear'] = 'nearest'
    """Pixel interpolation scheme used when sampling the sky map."""


@dataclass
class SotodlibConfig:
    """Configuration options specific to the sotodlib interface."""

    # see https://github.com/simonsobs/so3g/blob/master/python/proj/coords.py#L45
    site: Literal['so', 'so_sat1', 'so_sat2', 'so_sat3', 'so_lat'] = 'so'
    """Observatory site identifier"""

    weather: Literal['toco', 'typical'] = 'toco'
    """Atmospheric condition tag for so3g sightline model"""

    demodulated: bool = False
    """Use demodulated TODs (HWP-specific data from sotodlib preprocessing)."""

    wobble_correction: bool = False
    """Apply HWP wobble correction to the line of sight."""

    noise_source: Literal['preprocess', 'mapmaking'] = 'preprocess'
    """Which precomputed noise model to use when fit_noise_model is False.

    'preprocess': use per-stoke noise fits (noiseT, noiseQ, noiseU) from preprocessing.
    'mapmaking': use the white noise estimate from noiseQ_mapmaking.
    """


@dataclass
class MapMakingConfig:
    method: Methods = Methods.BINNED
    scanning_mask: bool = False
    sample_mask: bool = False
    hits_cut: float = 1e-2
    cond_cut: float = 1e-2
    double_precision: bool = True
    pointing: PointingConfig = field(default_factory=PointingConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    debug: bool = True
    solver: SolverConfig = field(default_factory=SolverConfig)
    gaps: GapsConfig = field(default_factory=GapsConfig)
    landscape: LandscapeConfig = field(
        default_factory=lambda: LandscapeConfig(healpix=HealpixConfig())
    )
    templates: TemplatesConfig | None = None
    atop_tau: int = 0
    sotodlib: SotodlibConfig | None = None

    @classmethod
    def full_defaults(cls) -> 'MapMakingConfig':
        """Create a config with default values for all fields including optional ones."""
        return cls(templates=TemplatesConfig.full_defaults())

    @classmethod
    def load_yaml(cls, path: str | Path) -> 'MapMakingConfig':
        """Load and instantiate a ``MapMakingConfig`` from a YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return cls.load_dict(data)

    @classmethod
    def load_dict(cls, data: dict[str, Any]) -> 'MapMakingConfig':
        """Load and instantiate a ``MapMakingConfig`` from a dictionary."""
        return deserialize(MapMakingConfig, data)

    def dump_yaml(self, path: str | Path) -> None:
        """Dump the config to a YAML file.

        The '.yaml' suffix is automatically added if not already present.
        """
        filename = Path(path).with_suffix('.yaml')
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(self._to_yaml())

    def _to_yaml(self) -> str:
        """Serialize the config to a YAML string."""
        data = serialize(MapMakingConfig, self)
        return yaml.dump(data, indent=2)

    @property
    def binned(self) -> bool:
        return self.noise.white

    @property
    def demodulated(self) -> bool:
        return self.sotodlib.demodulated if self.sotodlib is not None else False

    @property
    def use_templates(self) -> bool:
        return (self.templates is not None) and (not self.templates.empty)

    @property
    def dtype(self) -> DTypeLike:
        return jnp.float64 if self.double_precision else jnp.float32  # type: ignore[no-any-return]
