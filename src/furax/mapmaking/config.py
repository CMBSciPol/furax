from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NamedTuple

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


class WeightingMode(Enum):
    IDENTITY = 'identity'  # identity inverse-noise, no weighting
    DIAGONAL = 'diagonal'  # white (diagonal) weighting
    TOEPLITZ = 'toeplitz'  # atmospheric 1/f, banded Toeplitz weighting


class NoiseSource(Enum):
    FIT = 'fit'  # fit the noise model from the TOD PSD
    PRECOMPUTED = 'precomputed'  # read precomputed noise parameters from the data pipeline


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
class WeightingConfig:
    """Configuration for the inverse-noise / weighting matrix used in mapmaking.

    ``mode`` selects the structure of the inverse-noise matrix:

    - ``IDENTITY``: identity matrix (no weighting).
    - ``DIAGONAL``: diagonal white-noise weighting (default).
    - ``TOEPLITZ``: banded Toeplitz weighting for atmospheric (1/f) noise.

    ``source`` selects where the noise model comes from: ``FIT`` (default) fits it from the
    TOD power spectral density using ``fitting``; ``PRECOMPUTED`` reads noise parameters from
    the data pipeline (``fitting`` is then ignored).

    ``correlation_length`` sets the Toeplitz bandwidth (in samples) of the inverse-noise
    operator.  It is only used in ``TOEPLITZ`` mode and is ignored otherwise.
    """

    mode: WeightingMode = WeightingMode.DIAGONAL
    """Inverse-noise weighting matrix structure."""

    source: NoiseSource = NoiseSource.FIT
    """Where the noise model comes from: fit from the TOD PSD or read precomputed parameters."""

    correlation_length: int = 1_000
    """Toeplitz bandwidth in samples.  Only relevant in ``TOEPLITZ`` mode."""

    fitting: NoiseFitConfig = field(default_factory=NoiseFitConfig)
    """Options for fitting the noise PSD to the data.  Ignored when ``source`` is PRECOMPUTED."""

    @property
    def diagonal_matrix(self) -> bool:
        """True when the inverse-noise matrix is diagonal (identity or white)."""
        return self.mode != WeightingMode.TOEPLITZ


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


class PolynomialOrders(NamedTuple):
    """A polynomial order range, inclusive."""

    min_order: int = 0
    max_order: int = 3

    @property
    def n_orders(self) -> int:
        """Number of orders in the inclusive range."""
        return self.max_order - self.min_order + 1


@dataclass
class BinsConfig:
    """A piecewise basis that bins a variable into ``n_bins`` intervals.

    With ``interpolate = False`` each sample is hard-assigned to its bin. With
    ``interpolate = True`` samples are spread over neighbouring bin centres using
    triangular (or, if ``smooth``, sin^2) weights.
    """

    n_bins: int = 4
    interpolate: bool = False
    smooth: bool = False


@dataclass
class PolynomialConfig:
    legendre: PolynomialOrders = PolynomialOrders(0, 3)
    """Legendre orders for the polynomial drift template."""


@dataclass
class ScanSynchronousConfig:
    """Scan-synchronous signal on a global Legendre basis.

    Represents signals that depend only on the telescope's azimuth.
    """

    legendre: PolynomialOrders = PolynomialOrders(3, 7)


@dataclass
class BinAzSynchronousConfig:
    """Binned azimuth-synchronous signal, no HWP coupling.

    The binned counterpart of `ScanSynchronousConfig`.
    """

    bins: BinsConfig = field(default_factory=BinsConfig)


@dataclass
class HWPSynchronousConfig:
    n_harmonics: int = 3


@dataclass
class AzHWPSynchronousConfig:
    legendre: PolynomialOrders = PolynomialOrders(0, 3)
    n_harmonics: int = 4
    split_scans: bool = False


@dataclass
class BinAzHWPSynchronousConfig:
    bins: BinsConfig = field(default_factory=BinsConfig)
    n_harmonics: int = 4


@dataclass
class GroundConfig:
    azimuth_resolution: float = 0.05  # ~3 deg
    elevation_resolution: float = 0.05  # ~3 deg


@dataclass
class TemplatesConfig:
    polynomial: PolynomialConfig | None = None
    scan_synchronous: ScanSynchronousConfig | None = None
    binaz_synchronous: BinAzSynchronousConfig | None = None
    hwp_synchronous: HWPSynchronousConfig | None = None
    azhwp_synchronous: AzHWPSynchronousConfig | None = None
    binazhwp_synchronous: BinAzHWPSynchronousConfig | None = None
    ground: GroundConfig | None = None
    regularization: float = 0.0

    @classmethod
    def full_defaults(cls) -> 'TemplatesConfig':
        """Create a template config with default values for all templates."""
        return cls(
            polynomial=PolynomialConfig(),
            scan_synchronous=ScanSynchronousConfig(),
            binaz_synchronous=BinAzSynchronousConfig(),
            hwp_synchronous=HWPSynchronousConfig(),
            azhwp_synchronous=AzHWPSynchronousConfig(),
            binazhwp_synchronous=BinAzHWPSynchronousConfig(),
            ground=GroundConfig(),
        )

    @property
    def empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))


@dataclass(frozen=True)
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
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
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
    def for_method(cls, method: 'Methods | str') -> 'MapMakingConfig':
        """Return a default MapMakingConfig pre-configured for the given method.

        Args:
            method: A ``Methods`` enum value or its string name (e.g. ``'binned'``,
                ``'ml'``, ``'twostep'``, ``'atop'``), case-insensitive.
        """
        if isinstance(method, str):
            upper = method.upper()
            try:
                method = Methods[upper]
            except KeyError:
                # Fall back to matching by value (e.g. 'ML' → Methods.MAXL)
                matched = next((m for m in Methods if m.value.upper() == upper), None)
                if matched is None:
                    raise ValueError(f'Unknown method: {method!r}') from None
                method = matched

        if method == Methods.BINNED:
            return cls(
                method=Methods.BINNED,
                weighting=WeightingConfig(),
                templates=None,
            )
        elif method == Methods.MAXL:
            return cls(
                method=Methods.MAXL,
                weighting=WeightingConfig(
                    mode=WeightingMode.TOEPLITZ,
                    correlation_length=1_000,
                ),
                solver=SolverConfig(
                    rtol=1e-6,
                    atol=0,
                    max_steps=1_000,
                ),
                templates=None,
            )
        elif method == Methods.TWOSTEP:
            return cls(
                method=Methods.TWOSTEP,
                weighting=WeightingConfig(),
                solver=SolverConfig(
                    rtol=1e-6,
                    atol=0,
                    max_steps=1_000,
                ),
                templates=TemplatesConfig(
                    polynomial=PolynomialConfig(),
                ),
            )
        elif method == Methods.ATOP:
            return cls(
                method=Methods.ATOP,
                weighting=WeightingConfig(),
                solver=SolverConfig(
                    rtol=1e-6,
                    atol=0,
                    max_steps=100,
                ),
                atop_tau=37,
                templates=None,
            )
        else:
            raise ValueError(f'Unknown method: {method}')

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
        return self.weighting.diagonal_matrix

    @property
    def demodulated(self) -> bool:
        return self.sotodlib.demodulated if self.sotodlib is not None else False

    @property
    def use_templates(self) -> bool:
        return (self.templates is not None) and (not self.templates.empty)

    @property
    def dtype(self) -> DTypeLike:
        return jnp.float64 if self.double_precision else jnp.float32  # type: ignore[no-any-return]
