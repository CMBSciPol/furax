"""Hierarchical configuration system for mapmaking runs."""

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NamedTuple

import jax.numpy as jnp
import yaml
from apischema import deserialize, deserializer, serialize, serializer
from apischema.conversions import Conversion
from jax.typing import DTypeLike

from furax.obs.landscapes import ProjectionType
from furax.obs.stokes import ValidStokesLiteral

# apischema serializes IntEnum by value (integer) by default; override to use the name instead
# so that YAML config files show e.g. 'CAR' rather than '0'.
serializer(Conversion(lambda p: p.name, source=ProjectionType, target=str))
deserializer(Conversion(lambda s: ProjectionType[s], source=str, target=ProjectionType))

# Docs order: main config first, then its sub-configs grouped by topic.
__all__ = [
    'MapMakingConfig',
    'Methods',
    'PointingConfig',
    'WeightingConfig',
    'WeightingMode',
    'NoiseSource',
    'NoiseFitConfig',
    'SolverConfig',
    'GapsConfig',
    'GapTreatment',
    'GapFillingConfig',
    'NestedConfig',
    'LandscapeConfig',
    'HealpixConfig',
    'WCSConfig',
    'SkyPatch',
    'TemplatesConfig',
    'PolynomialOrders',
    'PolynomialConfig',
    'ScanSynchronousConfig',
    'BinsConfig',
    'BinAzSynchronousConfig',
    'HWPSynchronousConfig',
    'AzHWPSynchronousConfig',
    'BinAzHWPSynchronousConfig',
    'SplineHWPSSConfig',
    'T2PConfig',
    'GroundConfig',
    'SotodlibConfig',
]


class Methods(Enum):
    """Mapmaking algorithm selector (see [`MapMakingConfig.method`][])."""

    BINNED = 'Binned'
    """Invert the diagonal mapmaking system directly and skip the iterative solve.

    Only available when diagonal weights are used (see [`MapMakingConfig.weighting`][]).
    """

    MAXL = 'ML'
    """Classic maximum-likelihood mapmaking solve via conjugate gradient iteration."""

    ATOP = 'ATOP'
    """Polarisation (QU only) estimator using deprojection of short baselines.

    See [`MapMakingConfig.atop_tau`][].
    """


class WeightingMode(Enum):
    """Structure of the weighting matrix (see [`WeightingConfig.mode`][])."""

    IDENTITY = 'identity'
    """No weighting (identity matrix)."""

    DIAGONAL = 'diagonal'
    """Diagonal weighting (white noise model)."""

    TOEPLITZ = 'toeplitz'
    """Banded Toeplitz weighting ($1/f$ noise model)."""


class NoiseSource(Enum):
    """Where the noise model used for weighting comes from (see [`WeightingConfig.source`][])."""

    FIT = 'fit'
    """Fit the noise model from the TOD PSD."""

    PRECOMPUTED = 'precomputed'
    """Read precomputed noise parameters from the data pipeline."""


class GapTreatment(Enum):
    """How flagged samples enter the correlated-noise GLS weighting.

    For diagonal weights all three coincide ($M W M = M W$ for mask $M$ and diagonal $W$);
    the distinction only matters in the correlated regime.
    """

    INNER_MASK = 'inner_mask'
    """$W = M N^{-1} M$ (unbiased but suboptimal; cheap)."""

    FILL = 'fill'
    """Gap-fill the RHS with a constrained noise realization (single-realization use)."""

    NESTED = 'nested'
    """$W = M (M N M)^{-1} M$ (unbiased and minimum-variance)."""


@dataclass
class SolverConfig:
    """Options for the iterative (conjugate gradient) solver.

    Examples:
        YAML config section

            solver:
                rtol: 1.0e-6  # needs the decimal point, else PyYAML reads it as a string
                max_steps: 1000
    """

    rtol: float = 1e-6
    """Relative tolerance."""

    atol: float = 0
    """Absolute tolerance."""

    max_steps: int = 1_000
    """Maximum allowed number of iterations steps."""

    verbose: bool = False
    """Log the residual norm at every iteration."""

    @property
    def options(self) -> dict[str, Any]:
        """Dictionary of solver options, minus `verbose`."""
        return {k: v for k, v in asdict(self).items() if k != 'verbose'}


@dataclass
class NoiseFitConfig:
    r"""Options for fitting a parametric $1/f$ noise model to each detector's TOD PSD."""

    nperseg: int = 2_048
    """Welch window length in samples."""

    max_iter: int = 100
    """Maximum number of minimiser iterations."""

    tol: float = 1e-10
    """Relative minimiser tolerance (step size and function value change)."""

    min_freq_nyquist: float = 1e-8
    r"""Only use $f \geq$ `min_freq_nyquist` $\times f_\mathrm{Nyquist}$ for noise fitting."""

    max_freq_nyquist: float = 1
    r"""Only use $f <$ `max_freq_nyquist` $\times f_\mathrm{Nyquist}$ for noise fitting."""

    low_freq_nyquist: float = 0.02
    r"""PSD at $f <$ `low_freq_nyquist` $\times f_\mathrm{Nyquist}$ assumed dominated by $1/f$."""

    high_freq_nyquist: float = 0.02
    r"""PSD at $f >$ `high_freq_nyquist` $\times f_\mathrm{Nyquist}$ assumed white."""

    mask_hwp_harmonics: bool = True
    """Mask HWP harmonics: $1f$, $2f$, $4f$."""

    mask_ptc_harmonics: bool = False
    """Mask PTC harmonics: $1f$, $2f$."""

    freq_mask_width: float = 0.5
    """Full width of the frequency mask (if used) around HWP and PTC harmonics, in Hz."""

    ptc_freq: float = 1.4
    """PTC frequency used for masking (if used), in Hz."""


@dataclass
class WeightingConfig:
    """Configuration for the inverse-noise / weighting matrix used in mapmaking.

    Examples:
        Diagonal (white noise) weighting using precomputed noise parameters

            weighting:
                mode: diagonal
                source: precomputed

        Toeplitz ($1/f$) weighting

            weighting:
                mode: toeplitz
                correlation_length: 1000
    """

    mode: WeightingMode = WeightingMode.DIAGONAL
    """Matrix structure."""

    source: NoiseSource = NoiseSource.FIT
    """Where the noise model comes from."""

    correlation_length: int = 1_000
    """Toeplitz bandwidth in samples.  Only relevant in `TOEPLITZ` mode."""

    fitting: NoiseFitConfig = field(default_factory=NoiseFitConfig)
    """Options for fitting the noise PSD to the data."""

    @property
    def diagonal_matrix(self) -> bool:
        """True when the inverse-noise matrix is diagonal (identity or white)."""
        return self.mode != WeightingMode.TOEPLITZ


@dataclass
class HealpixConfig:
    """Configuration for a HEALPix output map."""

    nside: int = 512
    """HEALPix resolution parameter."""

    ordering: Literal['nest', 'ring'] = 'ring'
    """Pixel ordering scheme. Only ``'ring'`` is currently supported."""

    def __post_init__(self) -> None:
        if self.ordering == 'nest':
            raise ValueError('NESTED ordering not supported')


@dataclass
class SkyPatch:
    """Explicit rectangular sky patch for WCS map construction."""

    center: tuple[float, float]
    """Center ``(ra, dec)`` in degrees."""

    width: float
    """Width in degrees."""

    height: float
    """Height in degrees."""


@dataclass
class WCSConfig:
    """Configuration for a WCS-projected output map.

    The map extent is determined by exactly one of three mutually exclusive modes:

    - read from `geometry_file` (shape and WCS come from the file, all other fields ignored);
    - an explicit `patch` pixelised at `resolution`; or
    - automatic footprint detection from the observations (when neither is set), also pixelised
        at `resolution`.

    `projection` applies to the `patch`/auto modes only.
    """

    projection: ProjectionType = ProjectionType.CAR
    """WCS projection type."""

    resolution: float = 4.0
    """Pixel resolution in arcminutes."""

    geometry_file: str | None = None
    """Path to a FITS/HDF map file to read shape and WCS from via `pixell.enmap.read_map_geometry`."""

    patch: SkyPatch | None = None
    """Explicit sky patch definition."""

    def __post_init__(self) -> None:
        if self.geometry_file is not None and self.patch is not None:
            raise ValueError('geometry_file and patch are mutually exclusive.')

    @property
    def has_geometry(self) -> bool:
        """True if geometry is fixed by the configuration"""
        return self.geometry_file is not None or self.patch is not None


@dataclass
class LandscapeConfig:
    """Configuration of the output sky map: its Stokes components and pixelisation.

    Exactly one of `healpix` or `wcs` must be set.

    Examples:
        HEALPix output

            landscape:
                stokes: IQU
                healpix:
                    nside: 512

        WCS output, automatic footprint detection at 4 arcmin resolution

            landscape:
                stokes: IQU
                wcs:
                    resolution: 4.0

        WCS output, explicit patch

            landscape:
                stokes: IQU
                wcs:
                    resolution: 4.0
                    patch:
                        center: [30.0, -10.0]  # ra, dec in degrees
                        width: 20.0
                        height: 10.0

        WCS output, geometry read from a file

            landscape:
                stokes: IQU
                wcs:
                    geometry_file: /path/to/map.fits
    """

    stokes: ValidStokesLiteral = 'IQU'
    """Which Stokes components (`'I'`, `'QU'`, `'IQU'`, or `'IQUV'`) the output map holds."""

    healpix: HealpixConfig | None = None
    """HEALPix pixelisation. Mutually exclusive with ``wcs``."""

    wcs: WCSConfig | None = None
    """WCS (CAR) pixelisation. Mutually exclusive with ``healpix``."""

    def __post_init__(self) -> None:
        if (self.healpix is None) == (self.wcs is None):
            raise ValueError('exactly one of healpix or wcs must be set.')


class PolynomialOrders(NamedTuple):
    """A polynomial order range, inclusive."""

    min_order: int = 0
    """Lowest polynomial order in the range."""

    max_order: int = 3
    """Highest polynomial order in the range."""

    @property
    def n_orders(self) -> int:
        """Number of orders in the inclusive range."""
        return self.max_order - self.min_order + 1


@dataclass
class BinsConfig:
    """Configuration for binning a variable into `n_bins` intervals."""

    n_bins: int = 4
    """Number of bins."""

    interpolate: bool = False
    """Spread each sample over neighbouring bin centres instead of hard-assigning it to its bin.

    Interpolation uses triangular weights, or sin^2 if `smooth` is set.
    """

    smooth: bool = False
    """When `interpolate` is set, use sin^2 weights instead of triangular ones."""


@dataclass
class PolynomialConfig:
    """Polynomial drift template (per-detector low-order Legendre polynomial in time)."""

    legendre: PolynomialOrders = PolynomialOrders(0, 3)
    """Legendre orders for the polynomial drift template."""
    legendre_qu: PolynomialOrders | None = None
    """Legendre orders for the Q/U legs, demodulated data only.

    Overrides ``legendre`` for the Q and U legs (fitted independently from each other and
    from I). ``None`` reuses ``legendre`` for every leg. Requires ``demodulated=True``.
    """
    explicit: bool = False
    """If True, amplitudes are solved jointly and returned; if False, deprojected into W."""


@dataclass
class ScanSynchronousConfig:
    """Scan-synchronous signal on a global Legendre basis.

    Represents signals that depend only on the telescope's azimuth.
    """

    legendre: PolynomialOrders = PolynomialOrders(3, 7)
    """Legendre orders for the azimuth-dependent basis."""

    explicit: bool = False
    """If True, amplitudes are solved jointly and returned; if False, deprojected into W."""


@dataclass
class BinAzSynchronousConfig:
    """Binned azimuth-synchronous signal, no HWP coupling.

    The binned counterpart of [`ScanSynchronousConfig`][].
    """

    bins: BinsConfig = field(default_factory=BinsConfig)
    """Azimuth binning."""

    explicit: bool = False
    """If True, amplitudes are solved jointly and returned; if False, deprojected into W."""


@dataclass
class HWPSynchronousConfig:
    """HWP-synchronous signal on a global Fourier (harmonic) basis in HWP angle."""

    n_harmonics: int = 3
    """Number of HWP harmonics to fit."""

    explicit: bool = False
    """If True, amplitudes are solved jointly and returned; if False, deprojected into W."""


@dataclass
class AzHWPSynchronousConfig:
    """Joint azimuth/HWP-synchronous signal: Legendre in azimuth times Fourier in HWP angle."""

    legendre: PolynomialOrders = PolynomialOrders(0, 3)
    """Legendre orders for the azimuth-dependent basis."""

    n_harmonics: int = 4
    """Number of HWP harmonics to fit."""

    split_scans: bool = False
    """Fit independent coefficients per subscan instead of shared ones."""

    explicit: bool = False
    """If True, amplitudes are solved jointly and returned; if False, deprojected into W."""


@dataclass
class BinAzHWPSynchronousConfig:
    """Joint azimuth/HWP-synchronous signal: azimuth binning times Fourier in HWP angle.

    The binned-azimuth counterpart of [`AzHWPSynchronousConfig`][].
    """

    bins: BinsConfig = field(default_factory=BinsConfig)
    """Azimuth binning."""

    n_harmonics: int = 4
    """Number of HWP harmonics to fit."""

    explicit: bool = False
    """If True, amplitudes are solved jointly and returned; if False, deprojected into W."""


@dataclass
class SplineHWPSSConfig:
    """HWP-synchronous signal on a cubic B-spline basis in HWP angle."""

    n_knots: int | None = None
    """Number of spline knots. If set, takes precedence over `samples_per_knot`."""
    samples_per_knot: int | None = 4000
    """Number of samples per knot."""
    harmonics: tuple[int, ...] = (4,)
    """HWP harmonics to fit with splines."""

    explicit: bool = False
    """If True, amplitudes are solved jointly and returned; if False, deprojected into W."""

    def __post_init__(self) -> None:
        if self.n_knots is None and self.samples_per_knot is None:
            raise ValueError("one of 'n_knots' or 'samples_per_knot' must be provided")

    def resolve_n_knots(self, n_samples: int) -> int:
        """Number of spline knots for `n_samples` samples (at least 2).

        Uses `n_knots` directly when set, otherwise derives it from `samples_per_knot`.
        """
        if self.n_knots is not None:
            return max(2, self.n_knots)
        assert self.samples_per_knot is not None  # guaranteed by __post_init__
        return max(2, n_samples // self.samples_per_knot)


@dataclass
class T2PConfig:
    """Temperature-to-polarization leakage template (demodulated data only)."""

    fit_band: tuple[float, float] | None = None
    """Frequency band (in Hz) used to fit the leakage coefficients. `None` uses the full band."""

    decimate: int = 1
    """Decimation factor applied to the I template before fitting."""

    explicit: bool = True
    """T2P templates are always solved explicitly; deprojection is not supported."""

    def __post_init__(self) -> None:
        if not self.explicit:
            raise ValueError('T2P template filtering requires explicit=True')


@dataclass
class GroundConfig:
    """Ground pickup template: binned in (azimuth, elevation)."""

    azimuth_resolution: float = 0.05
    """Azimuth bin width in radians.

    Defaults to 0.05 (~3 deg).
    """

    elevation_resolution: float = 0.05
    """Elevation bin width in radians.

    Defaults to 0.05 (~3 deg).
    """

    explicit: bool = True
    """Ground templates are always solved explicitly; deprojection is not supported."""

    def __post_init__(self) -> None:
        if not self.explicit:
            raise ValueError('Ground template filtering requires explicit=True')


@dataclass
class TemplatesConfig:
    """Selection and configuration of templates to deproject/fit alongside sky map reconstruction.

    Each field is `None` by default, meaning that template is disabled; setting it to an instance
    of its config class enables the corresponding template with those options.

    Examples:
        YAML config section (enables the polynomial and ground templates)

            templates:
                polynomial:
                    legendre:
                        min_order: 0
                        max_order: 3
                ground:
                    azimuth_resolution: 0.05
    """

    polynomial: PolynomialConfig | None = None
    """Per-detector polynomial template."""

    scan_synchronous: ScanSynchronousConfig | None = None
    """Scan-synchronous (azimuth-only) template on a global Legendre basis."""

    binaz_synchronous: BinAzSynchronousConfig | None = None
    """Scan-synchronous (azimuth-only) template on a binned azimuth basis."""

    hwp_synchronous: HWPSynchronousConfig | None = None
    """HWP-synchronous template on a global Fourier (harmonic) basis."""

    azhwp_synchronous: AzHWPSynchronousConfig | None = None
    """Joint azimuth/HWP-synchronous template, Legendre in azimuth times Fourier in HWP angle."""

    binazhwp_synchronous: BinAzHWPSynchronousConfig | None = None
    """Joint azimuth/HWP-synchronous template, binned azimuth times Fourier in HWP angle."""

    spline_hwpss: SplineHWPSSConfig | None = None
    """HWP-synchronous template on a cubic B-spline basis in HWP angle."""

    t2p: T2PConfig | None = None
    """Temperature-to-polarization leakage template (demodulated data only)."""

    ground: GroundConfig | None = None
    """Ground pickup template, binned in (azimuth, elevation)."""

    regularization: float = 0.0
    """Ridge regularization strength applied to the template regression."""

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
            t2p=T2PConfig(),
            spline_hwpss=SplineHWPSSConfig(),
            ground=GroundConfig(),
        )

    @property
    def empty(self) -> bool:
        """True when every template is disabled."""
        return all(getattr(self, f.name) is None for f in fields(self))


@dataclass(frozen=True)
class GapFillingConfig:
    """Specific gap-filling options."""

    seed: int = 286502183
    """An integer seed for the noise realization"""

    max_steps: int = 50
    """The maximum number of iteration steps to invert the system"""

    rtol: float = 1e-4
    """The relative tolerance of the solver for the gap-filling solve"""

    precondition: bool = False
    """Precondition the flagged-subspace solve with the noise-model covariance (off by default)."""


@dataclass
class NestedConfig:
    """Inner-solver options for the nested-inverse gap weight (`GapTreatment.NESTED`)."""

    max_flag_fraction: float = 0.3
    """Flagged-fraction budget; observations flagged above this fall back to `INNER_MASK`."""

    inner_steps: int = 20
    """Maximum number of inner CG iterations for the flagged-block solve.

    Set `rtol = atol = 0` to force exactly this number of steps.
    """

    rtol: float = 0.0
    """Relative tolerance."""

    atol: float = 0.0
    """Absolute tolerance."""

    precondition: bool = False
    """Use the noise-model covariance to precondition the solve (off by default)."""


@dataclass
class GapsConfig:
    """Configuration options related to the treatment of gaps.

    Examples:
        Cheap, suboptimal inner-mask weighting (default correlated-noise fallback)

            gaps:
                treatment: inner_mask

        Gap-fill the RHS with a constrained noise realization

            gaps:
                treatment: fill
                fill_options:
                    seed: 42

        Unbiased, minimum-variance nested-inverse weighting

            gaps:
                treatment: nested
                nested:
                    inner_steps: 20
    """

    treatment: GapTreatment = GapTreatment.FILL
    """How flagged samples enter the weighting."""

    fill_options: GapFillingConfig = field(default_factory=GapFillingConfig)
    """Options to pass to the gap-filling operator (when `treatment` is `FILL`)."""

    nested: NestedConfig = field(default_factory=NestedConfig)
    """Inner-solver options (when `treatment` is `NESTED`)."""


@dataclass
class PointingConfig:
    """Configuration options for pointing computation.

    Pre-computed (`on_the_fly: false`) pointing with `interpolation: bilinear` caches the projected
    pixel coordinates and recovers interpolation weights on each apply; it requires a WCS/CAR
    landscape (HEALPix is not supported and raises at build time).

    Examples:
        Pre-computed pointing with bilinear sampling (WCS/CAR)

            pointing:
                on_the_fly: false
                interpolation: bilinear

        Pre-computed pointing, nearest-neighbor sampling (fastest)

            pointing:
                on_the_fly: false
    """

    on_the_fly: bool = True
    """Compute pointing on the fly instead of pre-computing pixel indices/coordinates."""

    batch_size: int = 32
    """Detector batch size for on-the-fly pointing (set to 0 to use a full batch)."""

    interpolation: Literal['nearest', 'bilinear'] = 'nearest'
    """Pixel interpolation scheme used when sampling the sky map.

    Pre-computed `'bilinear'` (`on_the_fly: false`) requires a WCS/CAR landscape.

    - ``'nearest'``: nearest-neighbor (default, fastest).
    - ``'bilinear'``: bilinear interpolation using the four nearest pixels.
    """


@dataclass
class SotodlibConfig:
    """Configuration options specific to the sotodlib interface.

    Examples:
        YAML config section

            sotodlib:
                site: so_sat1
                demodulated: true
    """

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
    """Precomputed noise model to use: preprocessing fits, or the mapmaking white-noise estimate."""


@dataclass
class MapMakingConfig:
    """Top-level configuration for a mapmaking run."""

    method: Methods = Methods.BINNED
    """Mapmaking algorithm to use."""

    scanning_mask: bool = False
    """Drop samples outside each observation's scanning intervals."""

    sample_mask: bool = False
    """Zero out TOD samples flagged invalid (glitches/cuts)."""

    hits_cut: float = 1e-2
    """Drop pixels whose hit count is below this threshold times the 95th-percentile hit count."""

    cond_cut: float = 1e-2
    """Drop pixels whose weight matrix rcond is smaller than this threshold."""

    double_precision: bool = True
    """Run the pipeline in float64 (`True`) or float32 (`False`); see [`dtype`][]."""

    pointing: PointingConfig = field(default_factory=PointingConfig)
    """Pointing computation options."""

    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    """Inverse-noise weighting / noise model options."""

    debug: bool = True
    """Run mapmaking solve twice to determine JIT time, and compute the reprojected map."""

    solver: SolverConfig = field(default_factory=SolverConfig)
    """Iterative (conjugate gradient) solver options."""

    gaps: GapsConfig = field(default_factory=GapsConfig)
    """Treatment of flagged samples in the correlated-noise weighting."""

    landscape: LandscapeConfig = field(
        default_factory=lambda: LandscapeConfig(healpix=HealpixConfig())
    )
    """Output sky map: Stokes components and pixelisation."""

    templates: TemplatesConfig | None = None
    """Template deprojection options. `None` disables all templates."""

    atop_tau: int = 0
    """Length of the `ATOP` interval (in samples)."""

    sotodlib: SotodlibConfig | None = None
    """Options specific to the sotodlib interface. `None` when not using sotodlib data."""

    def __post_init__(self) -> None:
        """Validate cross-field constraints that hold regardless of which mapmaker runs."""
        if (templates := self.templates) is not None:
            if templates.t2p is not None:
                if not self.demodulated:
                    raise ValueError('The T2P template requires demodulated=True.')
                if 'I' not in self.landscape.stokes:
                    raise ValueError(
                        "The T2P template requires an 'I' leg in landscape.stokes (got "
                        f'{self.landscape.stokes!r}).'
                    )
            if (
                templates.polynomial is not None
                and templates.polynomial.legendre_qu is not None
                and not self.demodulated
            ):
                raise ValueError('templates.polynomial.legendre_qu requires demodulated=True.')

    @classmethod
    def for_method(cls, method: 'Methods | str') -> 'MapMakingConfig':
        """Return a default MapMakingConfig pre-configured for the given method.

        Args:
            method: A ``Methods`` enum value or its string name (e.g. ``'binned'``,
                ``'ml'``, ``'atop'``), case-insensitive.
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
        """True when the inverse-noise weighting is diagonal (identity or white)."""
        return self.weighting.diagonal_matrix

    @property
    def demodulated(self) -> bool:
        """True when using demodulated TODs (sotodlib-specific)."""
        return self.sotodlib.demodulated if self.sotodlib is not None else False

    @property
    def use_templates(self) -> bool:
        """True when at least one template is enabled."""
        return (self.templates is not None) and (not self.templates.empty)

    @property
    def dtype(self) -> DTypeLike:
        """The floating-point dtype used throughout the pipeline, per `double_precision`."""
        return jnp.float64 if self.double_precision else jnp.float32  # type: ignore[no-any-return]
