"""Hierarchical configuration system for mapmaking runs."""

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NamedTuple, Self

import jax.numpy as jnp
import yaml
from jax.typing import DTypeLike

from furax.mapmaking._serialization import deserialize, serialize
from furax.obs.landscapes import ProjectionType
from furax.obs.stokes import ValidStokesLiteral

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
    'TwoLevelConfig',
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
    'SplineDriftConfig',
    'ScanSynchronousConfig',
    'BinsConfig',
    'BinnedAzimuthSynchronousConfig',
    'HWPSynchronousConfig',
    'AzimuthHWPSynchronousConfig',
    'BinnedAzimuthHWPSynchronousConfig',
    'SplineHWPSynchronousConfig',
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

    POMME = 'Pomme'
    """Polarisation (QU only) estimator using deprojection of short baselines.

    See [`MapMakingConfig.pomme_tau`][].
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
class TwoLevelConfig:
    """Options for the two-level (deflation) preconditioner.

    The eigenpairs of the system matrix (possibly preconditioned) are estimated by thick-restart
    Lanczos ([`lanczos_tr`][furax.linalg.lanczos_tr]) before the solve, and deflated by
    [`make_two_level_preconditioner`][furax.mapmaking.preconditioner.make_two_level_preconditioner].
    Each Lanczos step costs one product with the system matrix, like a CG iteration.

    Examples:
        YAML config section, under `solver`

            two_level:
                rank: 20
    """

    rank: int = 20
    """Number of eigenpairs to deflate."""

    which: Literal['small', 'large', 'both'] = 'small'
    """Which end of the spectrum to deflate.

    - `'small'`: the `rank` smallest eigenvalues.
    - `'large'`: the `rank` largest eigenvalues.
    - `'both'`: half of `rank` at each end, the extra one at the large end if `rank` is odd.

    Deflation pays at the end that sets the condition number, which depends on the system, e.g.
    on the noise model. The mapmaker logs the eigenvalue range of the deflated pairs.
    """

    spectrum: Literal['preconditioned', 'system'] = 'preconditioned'
    r"""Spectrum whose eigenvalues are deflated.

    - `'preconditioned'`: that of $M A$, the block-Jacobi preconditioned system.
    - `'system'`: that of $A$, the bare system operator.
    """

    krylov_size: int | None = None
    """Size of the Lanczos Krylov subspace, larger than `rank`. Defaults to `2 * rank`."""

    max_restarts: int = 20
    """Maximum number of Lanczos restart cycles."""

    tol: float = 1e-6
    """Relative tolerance on the Lanczos residual bound of each eigenpair."""

    seed: int = 0
    """Seed of the random Lanczos starting vector."""

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f'rank must be >= 1, got {self.rank}')


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

    two_level: TwoLevelConfig | None = None
    """Two-level preconditioner options. `None` uses the block-Jacobi preconditioner alone.

    Only the multi-observation mapmaker supports it.
    """

    @property
    def options(self) -> dict[str, Any]:
        """Dictionary of solver options, minus `verbose` and `two_level`."""
        return {k: v for k, v in asdict(self).items() if k not in ('verbose', 'two_level')}


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

    max_freq_nyquist: float = 1.0
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


def _per_stream_groups[T](values: T | dict[str, T], stokes: str) -> dict[str, T]:
    """The value of each group of streams fitted alike, keyed by its lowercase streams.

    Streams with the same value are grouped, in the order of `stokes`, whatever groups `values`
    names them in; streams it leaves out are absent. A plain value applies to every stream.

    Examples:
        >>> _per_stream_groups(PolynomialOrders(0, 3), 'IQU')
        {'iqu': PolynomialOrders(min_order=0, max_order=3)}
        >>> orders = PolynomialOrders(0, 1)
        >>> _per_stream_groups({'u': orders, 'Q': orders}, 'IQU')
        {'qu': PolynomialOrders(min_order=0, max_order=1)}
    """
    if not isinstance(values, dict):
        return {stokes.lower(): values}
    per_stream = {s: value for group, value in values.items() for s in group.lower()}
    groups: dict[T, str] = {}
    for s in stokes.lower():
        if (value := per_stream.get(s)) is not None:
            groups[value] = groups.get(value, '') + s
    return {group: value for value, group in groups.items()}


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
class SplineDriftConfig:
    """Slow drift of each detector, a cubic B-spline of time over the whole observation.

    Unlike a polynomial, the spline is local: each sample is seen by four neighbouring knots, so a
    flagged stretch leaves some knots unconstrained instead of making the template degenerate. The
    drift follows time scales longer than about twice the knot spacing.

    Examples:
        One knot every 560 samples, about 70 s at 8 Hz

            spline_drift:
                samples_per_knot: 560

        Demodulated data: slower drifts on I than on Q and U

            spline_drift:
                samples_per_knot:
                    i: 2000
                    qu: 560
    """

    samples_per_knot: int | dict[str, int] = 500
    """Samples between knots, for every stream or per group of streams of demodulated data.

    A knot count must be known before the timestamps are, so the spacing is in samples, not
    seconds. On a mapping such as `{'i': ..., 'qu': ...}` the streams of a group are fitted
    independently, and a stream left out has no spline template.
    """
    explicit: bool = False
    """If True, amplitudes are solved jointly and returned; if False, deprojected into W."""

    def __post_init__(self) -> None:
        values = (
            self.samples_per_knot.values()
            if isinstance(self.samples_per_knot, dict)
            else [self.samples_per_knot]
        )
        if any(value < 1 for value in values):
            raise ValueError(f'samples_per_knot must be >= 1, got {self.samples_per_knot}')


@dataclass
class ScanSynchronousConfig:
    """Scan-synchronous signal on a global Legendre basis.

    Represents signals that depend only on the telescope's azimuth.
    """

    legendre: PolynomialOrders | dict[str, PolynomialOrders] = field(
        default_factory=lambda: PolynomialOrders(3, 7)
    )
    """Legendre orders of the azimuth-dependent basis, for every stream or per group of streams.

    On demodulated data, a mapping such as `{'qu': ...}` fits the Q and U streams only, each
    independently; a stream left out has no scan-synchronous template.
    """

    explicit: bool = False
    """If True, amplitudes are solved jointly and returned; if False, deprojected into W."""


@dataclass
class BinnedAzimuthSynchronousConfig:
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
class AzimuthHWPSynchronousConfig:
    """Joint azimuth/HWP-synchronous signal: Legendre in azimuth times Fourier in HWP angle."""

    legendre: PolynomialOrders = field(default_factory=lambda: PolynomialOrders(0, 3))
    """Legendre orders for the azimuth-dependent basis."""

    n_harmonics: int = 4
    """Number of HWP harmonics to fit."""

    split_scans: bool = False
    """Fit independent coefficients per subscan instead of shared ones."""

    explicit: bool = False
    """If True, amplitudes are solved jointly and returned; if False, deprojected into W."""


@dataclass
class BinnedAzimuthHWPSynchronousConfig:
    """Joint azimuth/HWP-synchronous signal: azimuth binning times Fourier in HWP angle.

    The binned-azimuth counterpart of [`AzimuthHWPSynchronousConfig`][].
    """

    bins: BinsConfig = field(default_factory=BinsConfig)
    """Azimuth binning."""

    n_harmonics: int = 4
    """Number of HWP harmonics to fit."""

    explicit: bool = False
    """If True, amplitudes are solved jointly and returned; if False, deprojected into W."""


@dataclass
class SplineHWPSynchronousConfig:
    """HWP-synchronous signal whose harmonic amplitudes vary slowly with time.

    The template is built from harmonics of the HWP angle, each carried by a cubic B-spline in
    *time*: `n_knots` (or `samples_per_knot`) sets how fast the amplitude is allowed to vary.
    """

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
    r"""Temperature-to-polarization leakage template (demodulated data only).

    Each detector's Q and U pick up a fraction of its own temperature stream. By default that
    fraction is a free coefficient per detector and stream. Its amplitudes are always solved for
    with the map, never marginalized.

    With `zernike_order`, the coefficients instead follow the instrumental polarization of the
    optics, a polarization field over the focal plane shared by every detector. Off-axis rays
    cross the half-wave plate and the elements in front of it at an angle, which polarizes
    unpolarized light along the plane of incidence, increasingly with the field angle; a tilted or
    decentred element adds a polarization uniform across the field. Written in the focal-plane
    frame as $P(\xi, \eta) = \sum_k c_k Z_k(\xi / R, \eta / R)$ with complex $c_k$ and the Zernike
    polynomials $Z_k$ up to radial order `zernike_order`, it reaches detector $i$, of
    polarization angle $\gamma_i$, as the coefficient $e^{2i\gamma_i} P(\xi_i, \eta_i)$: its real
    part on Q and its imaginary part on U. A few amplitudes per observation replace a coefficient
    per detector.

    With `quadrupole_order` instead, the field is the one that optics symmetric about their axis
    produce. With $\zeta = (-\eta - i\xi) / R$ the complex position of a detector in the focal
    plane, the polarization makes exactly two turns around the axis, $P = \zeta^2 g(|\zeta|^2)$,
    and grows as the square of the field angle near the axis. `quadrupole_order` is the number of
    terms of the polynomial $g$ in $|\zeta|^2$. An axis shifted from the centre of the focal plane
    adds to the field, to first order in the shift, a polarization linear in $\zeta$ and a uniform
    one: `decentred` includes them. The field has `quadrupole_order` complex amplitudes, or
    three times as many if `decentred`.

    Examples:
        Leakage following the instrumental polarization of the optics, up to fourth order across
        the focal plane, fitted in the signal band

            t2p:
                fit_band: [0.03, 0.15]
                decimation_factor: 10
                zernike_order: 4

        The quadrupolar field of optics whose axis may be off the centre of the focal plane, to
        the first correction in the field angle: 6 complex amplitudes

            t2p:
                fit_band: [0.004, 0.1]
                quadrupole_order: 2
    """

    fit_band: tuple[float, float] | None = None
    """Frequency band (in Hz) used to fit the leakage coefficients. `None` uses the full band."""

    decimation_factor: int = 1
    """Decimation factor applied to the I template before fitting."""

    zernike_order: int | None = None
    """Highest radial order of the polarization field over the focal plane.

    `None` fits a free coefficient per detector instead.
    """

    quadrupole_order: int | None = None
    """Number of terms of the quadrupolar polarization field over the focal plane.

    `None` for no such field: the leakage is then a Zernike field if `zernike_order` is set, and a
    free coefficient per detector otherwise. Exclusive with `zernike_order`.
    """

    decentred: bool = True
    """Whether the quadrupolar field includes the terms of an axis shifted from the centre of the
    focal plane. Only with `quadrupole_order`."""

    zernike_radius: float = 0.31
    """Focal-plane radius $R$ in radians, scaling the detector coordinates of the field.

    The default covers the Simons Observatory SATs, whose detectors lie within 17.5 degrees of the
    boresight.
    """

    joint_qu: bool = True
    r"""Whether Q and U share one complex field, $\mathrm{Re}$ on Q and $\mathrm{Im}$ on U, or each
    fit their own. Only with `zernike_order` or `quadrupole_order`."""

    def __post_init__(self) -> None:
        if self.zernike_order is not None and self.zernike_order < 0:
            raise ValueError(f"'zernike_order' must be >= 0, got {self.zernike_order}")
        if self.quadrupole_order is not None and self.quadrupole_order < 1:
            raise ValueError(f"'quadrupole_order' must be >= 1, got {self.quadrupole_order}")
        if self.zernike_order is not None and self.quadrupole_order is not None:
            raise ValueError("'zernike_order' and 'quadrupole_order' are exclusive")
        if self.zernike_radius <= 0:
            raise ValueError(f"'zernike_radius' must be positive, got {self.zernike_radius}")


@dataclass
class CommonModeConfig:
    r"""Polarization common to the detectors, with a slow time dependence (demodulated data only).

    The window and the half-wave plate emit radiation that is partly polarized. All the detectors
    look through them, so they see the same polarization pattern across the focal plane, rising
    and falling together. The pattern is the quadrupolar field of [`T2PConfig`][], or any
    polynomial field with `polynomial_order`, and each of its modes has its own timestream, a cubic
    B-spline of time: a few spline coefficients per knot replace a signal in every detector. The amplitudes are shared by all the detectors of a run,
    so fit one band at a time. They are always solved for with the map, never marginalized.

    The uniform mode, equal for every detector, has the same effect as the polarization of the
    sky, so it removes sky signal at the time scales of the spline: it is off by default.

    Examples:
        The quadrupolar field and the term of a shifted axis, two complex amplitudes per knot, one
        knot every 10 s at 8 Hz

            common_mode:
                samples_per_knot: 80
    """

    samples_per_knot: int = 80
    """Samples between the knots of the time dependence. The amplitudes follow time scales longer
    than about twice the knot spacing."""

    quadrupole_order: int = 1
    """Number of terms of the quadrupolar field, as in [`T2PConfig`][]."""

    decentred: bool = True
    """Whether the field includes the terms of an axis shifted from the centre of the focal plane,
    as in [`T2PConfig`][]."""

    uniform: bool = False
    """Whether the field includes the polarization uniform across the focal plane."""

    polynomial_order: int | None = None
    r"""Highest degree of a polynomial field, in place of the quadrupolar one.

    The field is every monomial $\zeta^p \bar\zeta^q$ with $1 \le p + q \le$ `polynomial_order`
    (and the constant if `uniform`): $(n + 1)(n + 2) / 2 - 1$ complex amplitudes per knot, any
    smooth polarization pattern across the focal plane, not only the quadrupolar one. At order 2
    it holds the quadrupolar field and the term of a shifted axis. `quadrupole_order` and
    `decentred` are then ignored."""

    radius: float = 0.31
    """Focal-plane radius $R$ in radians, scaling the detector coordinates of the field."""

    def __post_init__(self) -> None:
        if self.samples_per_knot < 1:
            raise ValueError(f"'samples_per_knot' must be >= 1, got {self.samples_per_knot}")
        if self.quadrupole_order < 1:
            raise ValueError(f"'quadrupole_order' must be >= 1, got {self.quadrupole_order}")
        if self.polynomial_order is not None and self.polynomial_order < 1:
            raise ValueError(f"'polynomial_order' must be >= 1, got {self.polynomial_order}")
        if self.radius <= 0:
            raise ValueError(f"'radius' must be positive, got {self.radius}")


@dataclass
class GroundConfig:
    """Ground pickup template: binned in (azimuth, elevation), always solved for with the map."""

    azimuth_resolution: float = 0.05
    """Azimuth bin width in radians.

    Defaults to 0.05 (~3 deg).
    """

    elevation_resolution: float = 0.05
    """Elevation bin width in radians.

    Defaults to 0.05 (~3 deg).
    """


@dataclass
class TemplatesConfig:
    """Selection and configuration of templates to deproject/fit alongside sky map reconstruction.

    Each field is `None` by default, meaning that template is disabled; setting it to an instance
    of its config class enables the corresponding template with those options.

    Examples:
        YAML config section (enables the spline drift and ground templates)

            templates:
                spline_drift:
                    samples_per_knot: 560
                ground:
                    azimuth_resolution: 0.05
    """

    spline_drift: SplineDriftConfig | None = None
    """Per-detector drift template, a cubic B-spline of time over the observation."""

    scan_synchronous: ScanSynchronousConfig | None = None
    """Scan-synchronous (azimuth-only) template on a global Legendre basis."""

    binned_azimuth_synchronous: BinnedAzimuthSynchronousConfig | None = None
    """Scan-synchronous (azimuth-only) template on a binned azimuth basis."""

    hwp_synchronous: HWPSynchronousConfig | None = None
    """HWP-synchronous template on a global Fourier (harmonic) basis."""

    azimuth_hwp_synchronous: AzimuthHWPSynchronousConfig | None = None
    """Joint azimuth/HWP-synchronous template, Legendre in azimuth times Fourier in HWP angle."""

    binned_azimuth_hwp_synchronous: BinnedAzimuthHWPSynchronousConfig | None = None
    """Joint azimuth/HWP-synchronous template, binned azimuth times Fourier in HWP angle."""

    spline_hwp_synchronous: SplineHWPSynchronousConfig | None = None
    """HWP-synchronous template whose harmonic amplitudes follow a cubic B-spline in time."""

    t2p: T2PConfig | None = None
    """Temperature-to-polarization leakage template (demodulated data only)."""

    common_mode: CommonModeConfig | None = None
    """Polarization common to the detectors, with a slow time dependence (demodulated data only)."""

    ground: GroundConfig | None = None
    """Ground pickup template, binned in (azimuth, elevation)."""

    regularization: float = field(default=0.0, metadata={'template': False})
    """Ridge regularization strength applied to the template regression."""

    gram_batch_size: int = field(default=4, metadata={'template': False})
    """Number of detectors whose Grams are assembled together.

    Each detector's Gram takes temporary memory comparable to a few of its own timestreams, so this
    stays small; it is applied within each of [`MapMakingConfig.detector_batch_size`][]'s batches.
    """

    @classmethod
    def full_defaults(cls) -> Self:
        """Create a template config with default values for all templates."""
        return cls(
            spline_drift=SplineDriftConfig(),
            scan_synchronous=ScanSynchronousConfig(),
            binned_azimuth_synchronous=BinnedAzimuthSynchronousConfig(),
            hwp_synchronous=HWPSynchronousConfig(),
            azimuth_hwp_synchronous=AzimuthHWPSynchronousConfig(),
            binned_azimuth_hwp_synchronous=BinnedAzimuthHWPSynchronousConfig(),
            t2p=T2PConfig(),
            common_mode=CommonModeConfig(),
            spline_hwp_synchronous=SplineHWPSynchronousConfig(),
            ground=GroundConfig(),
        )

    @property
    def empty(self) -> bool:
        """True when every template is disabled."""
        # a field counts as a template unless marked otherwise
        return all(
            getattr(self, field.name) is None
            for field in fields(self)
            if field.metadata.get('template', True)
        )


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

    rtol: float = 0
    """Relative tolerance."""

    atol: float = 0
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

    batch_samples: int | None = None
    """Number of samples per batch of on-the-fly pointing.

    This is an advanced option. See
    [`PointingOperator.batch_samples`][furax.obs.pointing.PointingOperator.batch_samples].
    """

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

    downsample: int = 1
    """Integer factor by which each observation is downsampled (useful with `demodulated`).

    Every sample-axis field is downsampled after loading; timestreams are Fourier-resampled to
    avoid aliasing.
    """

    wobble_correction: bool = False
    """Apply HWP wobble correction to the line of sight."""

    noise_source: Literal['preprocess', 'mapmaking'] = 'preprocess'
    """Precomputed noise model to use: preprocessing fits, or the mapmaking white-noise estimate."""

    def __post_init__(self) -> None:
        if self.downsample < 1:
            raise ValueError(f'downsample must be >= 1, got {self.downsample}')


class InjectionMode(Enum):
    """How a simulated sky enters a run (see [`InjectionConfig.mode`][])."""

    ADD = 'add'
    """Add the sky to the TOD of every observation, before anything is derived from them."""

    REPLACE = 'replace'
    """Replace the data by the sky: no noise is left, and the weights must not be fit from them."""

    RESPONSE = 'response'
    """Leave the data alone and solve for the map of the sky alongside the map of the data."""


@dataclass
class InjectionConfig:
    """A simulated sky scanned into the data of a run, to measure what the run does to a signal.

    The signal is the sky seen through the acquisition operator of each observation: its pointing,
    its detectors' polarization angles and, for modulated data, the half-wave plate.
    """

    path: str
    """FITS file of the sky, on the pixelization of the landscape (a WCS map, or a HEALPix map in
    the landscape's ordering). It is written by a run that saves maps, so one can be a template."""

    stokes: ValidStokesLiteral = 'IQU'
    """The Stokes components stored in the file, in order; the others are zero. The temperature
    is needed for the T2P template, which is built from it, in the modes that add it to the TOD."""

    scale: float = 1.0
    """Factor applied to the sky."""

    mode: InjectionMode = InjectionMode.ADD
    """How the sky enters the run.

    The map is linear in the data once the weights, the pointing and the templates are fixed.
    With `response`, they all come from the data alone, and the sky is only a second right-hand
    side of the normal system, solved by a second CG: the result is the map of the sky through the
    run, `MapMakingResults.injected_map`, without another pass over the data.

    `add` also lets the sky into what is derived from the TOD. The T2P basis, built from the I
    stream, then carries the sky's temperature, and a template can absorb the polarization that
    correlates with it: a loss of signal that `response` does not show, since the basis of the data
    does not carry the sky. The map of the data with the sky added, minus the map of the data
    alone, is the same as the response where no template is built from the TOD.
    """


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
    """Run the pipeline in float64 (`True`) or float32 (`False`); see [`dtype`][].

    The multi-observation mapmaker always reads the TOD in float32, and converts each detector
    batch to [`dtype`][] before using it.
    """

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

    pomme_tau: int = 0
    """Length of the `Pomme` interval (in samples)."""

    max_buckets: int = 4
    """Largest number of buckets observations are grouped into.

    More buckets pad the buffers less but round up to the device count more often; see
    [`furax.mapmaking.layout`][] for how to choose it.
    """

    detector_batch_size: int = 16
    """Size of detector batches.

    Using detector batches reduces peak memory: beyond the TOD itself, everything computed from it
    only takes memory in proportion to the batch rather than to the whole observation. Detector
    counts are padded to whole batches. A size of 0 processes all detectors at once (required for
    gap-filling and nested gap treatment with a non-diagonal noise model).
    """

    sotodlib: SotodlibConfig | None = None
    """Options specific to the sotodlib interface. `None` when not using sotodlib data."""

    injection: InjectionConfig | None = None
    """A simulated sky to add to the data, or to replace them with. `None` for none."""

    def __post_init__(self) -> None:
        """Validate cross-field constraints that hold regardless of which mapmaker runs."""
        if self.max_buckets < 1:
            raise ValueError(f'max_buckets must be >= 1, got {self.max_buckets}')
        if self.detector_batch_size < 0:
            raise ValueError(f'detector_batch_size must be >= 0, got {self.detector_batch_size}')
        if (
            self.detector_batch_size
            and not self.binned
            and self.gaps.treatment in (GapTreatment.FILL, GapTreatment.NESTED)
        ):
            raise ValueError(
                f'the {self.gaps.treatment.value} gap treatment works on whole observations: set '
                f'detector_batch_size to 0 (got {self.detector_batch_size})'
            )
        if (injection := self.injection) is not None:
            if (
                injection.mode == InjectionMode.REPLACE
                and self.weighting.mode != WeightingMode.IDENTITY
                and self.weighting.source == NoiseSource.FIT
            ):
                raise ValueError(
                    'injection.mode=replace removes the noise from the data, so the weights cannot '
                    'be fit from them: set weighting.source to precomputed.'
                )
            if (
                injection.mode == InjectionMode.RESPONSE
                and self.gaps.treatment == GapTreatment.FILL
                and not self.binned
            ):
                raise ValueError(
                    'injection.mode=response needs a map linear in the data: gap filling is not.'
                )
        if (templates := self.templates) is not None:
            for label, polarized in (
                ('T2P', templates.t2p),
                ('common-mode', templates.common_mode),
            ):
                if polarized is None:
                    continue
                if not self.demodulated:
                    raise ValueError(f'The {label} template requires demodulated=True.')
                if 'QU' not in self.landscape.stokes:
                    raise ValueError(
                        f'The {label} template requires Q and U streams in landscape.stokes (got '
                        f'{self.landscape.stokes!r}).'
                    )
            for name, option in (
                ('scan_synchronous', 'legendre'),
                ('spline_drift', 'samples_per_knot'),
            ):
                template = getattr(templates, name)
                if template is not None and isinstance(
                    per_stream := getattr(template, option), dict
                ):
                    self._check_stream_groups(f'templates.{name}.{option}', per_stream)

    def _check_stream_groups(self, name: str, per_stream: dict[str, Any]) -> None:
        """Check that a template option per stream names each stream of the landscape at most once."""
        if not self.demodulated:
            raise ValueError(f'{name} per stream requires demodulated=True.')
        streams = ''.join(per_stream).upper()
        if not streams:
            raise ValueError(f'{name} names no stream.')
        if extra := sorted(set(streams) - set(self.landscape.stokes)):
            raise ValueError(
                f'{name} names streams {extra} outside landscape.stokes={self.landscape.stokes!r}.'
            )
        if repeated := sorted({s for s in streams if streams.count(s) > 1}):
            raise ValueError(f'{name} names streams {repeated} more than once.')

    @classmethod
    def for_method(cls, method: 'Methods | str') -> Self:
        """Return a default MapMakingConfig pre-configured for the given method.

        Args:
            method: A ``Methods`` enum value or its string name (e.g. ``'binned'``,
                ``'ml'``, ``'pomme'``), case-insensitive.
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
                detector_batch_size=0,  # the default gap filling works on whole observations
            )
        elif method == Methods.POMME:
            return cls(
                method=Methods.POMME,
                weighting=WeightingConfig(),
                solver=SolverConfig(
                    rtol=1e-6,
                    atol=0,
                    max_steps=100,
                ),
                pomme_tau=37,
                templates=None,
            )
        else:
            raise ValueError(f'Unknown method: {method}')

    @classmethod
    def load_yaml(cls, path: str | Path) -> Self:
        """Load and instantiate a ``MapMakingConfig`` from a YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return cls.load_dict(data)

    @classmethod
    def load_dict(cls, data: dict[str, Any]) -> Self:
        """Load and instantiate a ``MapMakingConfig`` from a dictionary."""
        return deserialize(cls, data)

    def dump_yaml(self, path: str | Path) -> None:
        """Dump the config to a YAML file.

        The '.yaml' suffix is automatically added if not already present.
        """
        filename = Path(path).with_suffix('.yaml')
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(self._to_yaml())

    def _to_yaml(self) -> str:
        """Serialize the config to a YAML string."""
        data = serialize(self)
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
    def read_stokes(self) -> ValidStokesLiteral:
        """The demodulated streams read: those of the map, and I for the T2P template.

        The T2P template is built from the I timestream, which is read even when the map has no I.
        """
        t2p = self.templates is not None and self.templates.t2p is not None
        return 'IQU' if t2p and self.landscape.stokes == 'QU' else self.landscape.stokes

    @property
    def dtype(self) -> DTypeLike:
        """The floating-point dtype used throughout the pipeline, per `double_precision`."""
        return jnp.float64 if self.double_precision else jnp.float32
