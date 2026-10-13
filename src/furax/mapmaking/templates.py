"""Template operators for fitting structured nuisance signals out of the data.

A template turns a small set of per-detector amplitudes into a time stream, so the mapmaker can fit
and remove unwanted but predictable signals (slow drifts, scan-synchronous pickup, HWP-synchronous
lines, T-to-P leakage).

Each template is described by a [`Basis`][furax.mapmaking.bases.Basis] of functions of time, built
here from the observation's data (`polynomial_basis`, `t2p_basis`, ...). A
[`TemplateOperator`][] combines every enabled template into one operator, keyed by template name,
giving each detector its own amplitudes; [`StokesTemplateOperator`][] does the same for a
[`Stokes`][] TOD, one amplitude set per leg.
"""

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import field
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import DTypeLike, Float, PyTree

from furax import AbstractLinearOperator
from furax.core import TransposeOperator
from furax.math import bspline
from furax.obs.stokes import Stokes, ValidStokesLiteral

from .bases import Basis, KroneckerBasis, SegmentedBasis, TensorBasis, WindowedBasis, _einsum
from .config import BinsConfig, PolynomialOrders

__all__ = [
    'StokesLeg',
    'polynomial_basis',
    't2p_basis',
    'scan_synchronous_basis',
    'binned_azimuth_synchronous_basis',
    'hwp_synchronous_basis',
    'azimuth_hwp_synchronous_basis',
    'binned_azimuth_hwp_synchronous_basis',
    'spline_hwp_synchronous_basis',
    'AbstractTemplateOperator',
    'TemplateOperator',
    'StokesTemplateOperator',
]


StokesLeg = Literal['i', 'q', 'u', 'v']
"""One Stokes component, lower case."""


def _bin_weights(
    x: Float[Array, ' samp'],
    n_bins: int,
    interpolate: bool,
    smooth: bool,
    dtype: DTypeLike,
) -> Float[Array, '{n_bins} samp']:
    """Assign each sample to a bin of `x`.

    Splits the range of `x` into `n_bins` equal bins. Entry `[j, s]` is how much sample `s` belongs
    to bin `j`. With `interpolate=False` this is one-hot: 1 for the bin the sample falls in, 0
    elsewhere. With `interpolate=True` a sample near a bin edge is shared with its neighbour
    (triangular weights, or sin² hats if `smooth`), each sample's weights summing to 1.

    Each row is then one basis function of a binned template (a bin's time profile).
    """
    n_samps = x.size
    lo = jnp.min(x)
    hi = jnp.max(x) + 1e-8  # nudge so the global max falls in the last bin
    edges = jnp.linspace(lo, hi, n_bins + 1)

    if not interpolate:
        # Hard assignment: each sample contributes 1.0 to exactly its bin.
        sample_bin = jnp.digitize(x, edges[1:])
        bins = jnp.zeros((n_bins, n_samps)).at[sample_bin, jnp.arange(n_samps)].set(1.0)
        return bins.astype(dtype)

    # Soft assignment: triangular weights peaking at each bin centre, falling
    # linearly to zero one bin-width (`delta`) away. Shape (n_bins, n_samps).
    centres = 0.5 * (edges[:-1] + edges[1:])
    delta = (hi - lo) / n_bins
    triangular = jnp.clip(1 - jnp.abs(x[None, :] - centres[:, None]) / delta, min=0)

    if smooth:
        # sin^2 reshaping of the triangle, renormalised so weights sum to 1 per sample.
        bins = jnp.sin((jnp.pi / 2) * triangular) ** 2
        bins /= jnp.sum(bins, axis=0)[None, :]
    else:
        # Linear interpolation; clamp samples beyond the end centres to the edge bins.
        bins = triangular.at[0, x < centres[0]].set(1.0).at[-1, x > centres[-1]].set(1.0)

    return bins.astype(dtype)


def _legendre_values(
    u: Float[Array, ' samp'],
    min_order: int,
    max_order: int,
    dtype: DTypeLike,
) -> Float[Array, '{max_order-min_order+1} samp']:
    """Legendre polynomials of orders `min_order..max_order` (inclusive).

    Evaluated on `u`, already rescaled to `[-1, 1]`.
    """
    legs = jax.scipy.special.lpmn_values(max_order, max_order, u, is_normalized=False)
    return legs[0, min_order:, :].astype(dtype)


def _legendre(
    x: Float[Array, ' samp'],
    min_order: int,
    max_order: int,
    dtype: DTypeLike,
) -> Float[Array, '{max_order-min_order+1} samp']:
    """Legendre polynomials of orders `min_order..max_order` (inclusive).

    Evaluated on `x`, rescaled to `[-1, 1]` over its global range.
    """
    u = -1.0 + 2.0 * (x - jnp.min(x)) / jnp.ptp(x)
    return _legendre_values(u, min_order, max_order, dtype)


def _harmonics(
    angles: Float[Array, ' samp'],
    harmonics: int | Sequence[int],
    dtype: DTypeLike,
    *,
    dc: bool,
) -> Float[Array, ' rows samp']:
    """Harmonic basis `sin(k·angle), cos(k·angle)` for each harmonic `k`.

    Optionally prepended with a constant (DC) row when `dc` is set.

    `harmonics` is either an int `n` (the harmonics `1..n`) or an explicit sequence of harmonic
    orders.
    """
    h = jnp.arange(1, harmonics + 1) if isinstance(harmonics, int) else jnp.asarray(harmonics)
    sines = jnp.sin(h[:, None] * angles[None, :])
    cosines = jnp.cos(h[:, None] * angles[None, :])
    parts = ([jnp.ones((1, angles.size), dtype=dtype)] if dc else []) + [sines, cosines]
    return jnp.concatenate(parts, axis=0).astype(dtype)


def polynomial_basis(
    max_poly_order: int,
    intervals: Float[Array, 'n_intervals 2'],
    times: Float[Array, ' samp'],
    dtype: DTypeLike,
    valid_mask: Float[Array, ' samp'] | None = None,
    min_poly_order: int = 0,
) -> Basis:
    """Basis for a polynomial drift template, one polynomial per scanning interval.

    Each sample belongs to one interval and is fitted with Legendre orders
    `min_poly_order..max_poly_order` over that interval. Under Pomme the constant is already
    removed, so `min_poly_order=1` keeps the basis non-degenerate.

    Assumes `intervals` are sorted, non-overlapping `[start, end)` rows. Samples in gaps or past the
    last interval get a zero basis column. `valid_mask` optionally zeroes flagged samples (1 = keep,
    0 = drop) so they neither carry template signal nor constrain the fitted amplitudes.
    """
    n_samps = times.size
    n_intervals = intervals.shape[0]
    starts = intervals[:, 0]
    ends = intervals[:, 1]

    s = jnp.arange(n_samps)
    # interval id per sample: last interval whose start <= s (intervals sorted),
    # clamped into range. Gaps/out-of-range are caught by `in_range` below.
    segment = jnp.clip(jnp.searchsorted(starts, s, side='right') - 1, 0, n_intervals - 1)
    seg_start = starts[segment]
    seg_end = ends[segment]
    in_range = (s >= seg_start) & (s < seg_end)

    t0 = times[seg_start]
    span = jnp.where(seg_end > seg_start + 1, times[seg_end - 1] - t0, 1.0)
    # rescale each sample to [-1, 1] within its own interval; out-of-range
    # samples sit at 0 and are zeroed by `in_range` below.
    u = jnp.where(in_range, -1.0 + 2.0 * (times - t0) / span, 0.0)
    legs = _legendre_values(u, min_poly_order, max_poly_order, dtype)  # (k, n_samps)
    legs = legs * in_range[None, :]
    if valid_mask is not None:
        legs = legs * valid_mask[None, :].astype(dtype)

    return SegmentedBasis(segment.astype(jnp.int32), legs, n_intervals)


_T2P_BATCH_SIZE = 32
"""Detectors band-passed at once by [`t2p_basis`][]."""


def t2p_basis(
    temperature: Float[Array, 'det samp'],
    dtype: DTypeLike,
    fit_band: tuple[float, float] | None = None,
    sample_rate: Float[Array, ''] | float = 1.0,
    decimation_factor: int = 1,
) -> Basis:
    """Per-detector basis for a temperature-to-polarization leakage template.

    Each detector's basis is just its own temperature stream, so fitting one amplitude per detector
    estimates how much temperature leaks into its polarization.

    `fit_band=(f0, f1)` restricts the temperature basis to that frequency band (Hz), so the leakage
    is both estimated and removed only there, keeping the template a clean linear operator.

    `decimation_factor=q` stores the basis on a `q`-times coarser grid to cut memory; the
    coarse-grid Nyquist frequency `sample_rate / 2q` must stay above `f1`. As a rule of thumb keep
    it at a few times `f1` (`q ≲ sample_rate / 6·f1`): the fractional error on the fitted amplitude
    grows like `(f1 / (sample_rate / 2q))²`.

    Assumes `temperature` is already deglitched/gap-filled upstream: a glitch left in it would smear
    across the band and bias the fitted amplitude.
    """
    n_full = temperature.shape[-1]
    q = decimation_factor
    n_dec = -(-n_full // q)  # ceil

    band = None
    if fit_band is not None:
        f0, f1 = fit_band
        freqs = jnp.fft.rfftfreq(n_full, d=1.0 / sample_rate)
        band = (freqs > f0) & (freqs < f1)

    def filter_and_decimate(t: Float[Array, ' samp']) -> Float[Array, ' dec']:
        if band is not None:
            t = jnp.fft.irfft(jnp.fft.rfft(t) * band, n=n_full)
        if q > 1:
            # Block-average onto a q-times coarser grid: pad the tail to a whole block,
            # reshape (n_dec, q) and mean. `TensorBasis` hold-upsamples back to `n_full` in
            # synthesis (band-limits above sample_rate / 2q).
            t = jnp.pad(t, (0, n_dec * q - n_full)).reshape(n_dec, q).mean(axis=-1)
        return t.astype(dtype)

    t = jax.lax.map(filter_and_decimate, temperature, batch_size=_T2P_BATCH_SIZE)
    values = t[:, None, :]  # (det, k=1, dec)
    if q > 1:
        return TensorBasis.per_detector_stack(values=values, q=q, n_full=n_full)
    return TensorBasis.per_detector_stack(values=values)


def scan_synchronous_basis(
    legendre: PolynomialOrders,
    azimuth: Float[Array, ' samp'],
    dtype: DTypeLike,
) -> Basis:
    """Scan-synchronous (azimuth-only) basis on a global Legendre basis."""
    legs = _legendre(azimuth, legendre.min_order, legendre.max_order, dtype)
    return TensorBasis(legs)


def binned_azimuth_synchronous_basis(
    bins: BinsConfig,
    azimuth: Float[Array, ' samp'],
    dtype: DTypeLike,
) -> Basis:
    """Binned azimuth-synchronous basis, no HWP coupling: one amplitude per azimuth bin."""
    weights = _bin_weights(azimuth, bins.n_bins, bins.interpolate, bins.smooth, dtype)
    return TensorBasis(weights)


def hwp_synchronous_basis(
    n_harmonics: int,
    hwp_angles: Float[Array, ' samp'],
    dtype: DTypeLike,
) -> Basis:
    """HWP-synchronous basis: harmonics of the HWP angle, `k = 1..n_harmonics`."""
    matrix = _harmonics(hwp_angles, n_harmonics, dtype, dc=False)
    return TensorBasis(matrix)


def azimuth_hwp_synchronous_basis(
    legendre: PolynomialOrders,
    n_harmonics: int,
    azimuth: Float[Array, ' samp'],
    hwp_angles: Float[Array, ' samp'],
    dtype: DTypeLike,
    scan_mask: Float[Array, ' samp'] | None = None,
) -> Basis:
    """Azimuth-Legendre x HWP-harmonic basis (Kronecker product of the two).

    `scan_mask` optionally zeroes the azimuth leg on flagged samples (e.g. to fit separate
    amplitudes per scan direction).
    """
    poly = _legendre(azimuth, legendre.min_order, legendre.max_order, dtype)
    if scan_mask is not None:
        poly = scan_mask[None, :] * poly
    harm = _harmonics(hwp_angles, n_harmonics, dtype, dc=True)
    return KroneckerBasis((poly, harm))


def binned_azimuth_hwp_synchronous_basis(
    bins: BinsConfig,
    n_harmonics: int,
    azimuth: Float[Array, ' samp'],
    hwp_angles: Float[Array, ' samp'],
    dtype: DTypeLike,
) -> Basis:
    """Azimuth-binned x HWP-harmonic basis (azimuth is always binned)."""
    bin_basis = _bin_weights(azimuth, bins.n_bins, bins.interpolate, bins.smooth, dtype)
    harm = _harmonics(hwp_angles, n_harmonics, dtype, dc=True)
    return KroneckerBasis((bin_basis, harm))


def spline_hwp_synchronous_basis(
    times: Float[Array, ' samp'],
    hwp_angles: Float[Array, ' samp'],
    n_knots: int,
    harmonics: int | Sequence[int],
    dtype: DTypeLike,
) -> Basis:
    """Spline-based HWP synchronous basis.

    A cubic B-spline models the slowly time-varying amplitude of the HWP-synchronous signal: knot
    `j` carries a `(sin kχ, cos kχ)` pair for each harmonic `k`, so the amplitudes have shape `(K,
    2*n_harmonics)` with `K = n_knots + 2`.

    Assumes `times` is non-decreasing.
    """
    offset, weights = bspline.spline_window(times, n_knots)  # weights (samp, 4)
    sub_values = _harmonics(hwp_angles, harmonics, dtype, dc=False).astype(dtype)
    return WindowedBasis(offset, weights.T.astype(dtype), sub_values, n_blocks=n_knots + 2)


def is_basis(x: Any) -> bool:
    return isinstance(x, Basis)


class AbstractTemplateOperator(AbstractLinearOperator):
    """Every enabled template as one operator, from amplitudes keyed by template name to a TOD."""

    # The bases must stay the only dynamic leaves: an operator then stacks under a scan over
    # observations by gaining a leading axis on the basis arrays, everything else being static.
    bases: dict[str, Any]
    n_dets: int = field(metadata={'static': True})

    def __post_init__(self) -> None:
        # `in_structure` is derived, never passed: it can then never disagree with the bases.
        if self.in_structure is not None:
            raise ValueError('in_structure is derived from the bases; do not pass it')
        object.__setattr__(self, 'in_structure', self._amplitude_structure())
        super().__post_init__()

    # ---- amplitude side -------------------------------------------------------------------------
    def _amplitude_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        """`bases` with each basis replaced by the amplitudes it takes."""
        return jax.tree.map(self._amplitude_leaf, self.bases, is_leaf=is_basis)

    def _amplitude_leaf(self, basis: Basis) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.n_dets, *basis.shape), basis.dtype)

    @property
    def bases_by_leg(self) -> dict[str, Any]:
        """`bases` keyed as the amplitudes are: a basis shared by several legs appears once per leg."""
        return self.bases

    # ---- TOD side -------------------------------------------------------------------------------
    def _a_basis(self) -> Basis:
        """Any one of the bases: they agree on sample count and dtype."""
        return jax.tree.leaves(self.bases, is_leaf=is_basis)[0]

    def _stream_structure(self) -> jax.ShapeDtypeStruct:
        """One detector-by-sample stream, the shape every Stokes leg of the TOD takes."""
        b = self._a_basis()
        return jax.ShapeDtypeStruct((self.n_dets, b.n_points), b.dtype)

    def _zero_stream(self) -> Array:
        s = self._stream_structure()
        return jnp.zeros(s.shape, s.dtype)

    def _expand_stream(self, bases: dict[str, Basis], x: dict[str, Array]) -> Array:
        """The summed expansion of every template on one stream."""
        # Each template would otherwise make its own pass over the stream. The dense ones are
        # instead stacked into one `(K, samp)` matrix, applied in one product over all detectors.
        dense = {n: v for n, basis in bases.items() if (v := basis._dense_values()) is not None}
        stream = self._zero_stream()
        if dense:
            coeffs = jnp.concatenate([x[name].reshape(self.n_dets, -1) for name in dense], axis=1)
            stream = _einsum('dk,ks->ds', coeffs, jnp.concatenate(list(dense.values())))
        for name, basis in bases.items():
            if name not in dense:
                stream = stream + basis._expand_detectors(x[name])
        return stream

    # ---- adjoint --------------------------------------------------------------------------------
    @abstractmethod
    def project(self, tod: PyTree[Array]) -> PyTree[Array]:
        """Every template's amplitudes, from a TOD. This is the transpose's `mv`."""

    def transpose(self) -> AbstractLinearOperator:
        return _TemplateOperatorTranspose(self)


class TemplateOperator(AbstractTemplateOperator):
    """Templates over a TOD with no Stokes axis, producing one `(n_dets, n_points)` array.

    Each template contributes `(n_dets, *basis.shape)` amplitudes, and the TOD is their summed
    expansion.
    """

    bases: dict[str, Basis]

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._stream_structure()

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        return self._expand_stream(self.bases, x)

    def project(self, tod: PyTree[Array]) -> PyTree[Array]:
        return {name: basis._project_detectors(tod) for name, basis in self.bases.items()}


class StokesTemplateOperator(AbstractTemplateOperator):
    """Templates over a [`Stokes`][] TOD, each leg one `(n_dets, n_points)` array.

    A Stokes-valued TOD carries one differently filtered stream per leg (as demodulation produces),
    so a template fits an independent set of amplitudes on each leg it covers. It need not cover
    them all: temperature-to-polarization leakage is fitted on Q and U only.

    Each template's bases are keyed by leg group: a key such as `'qu'` shares one basis between
    those legs, stored once, while each leg keeps its own amplitudes. The amplitudes are keyed by
    single leg whatever the grouping.

    Raises:
        TypeError: If a template is given as a bare basis rather than keyed by leg.
        ValueError: If a template names a leg outside `stokes`, or names one twice.

    Examples:
        >>> poly = TensorBasis(jnp.ones((2, 5)))
        >>> leak = TensorBasis(jnp.ones((1, 5)))
        >>> op = StokesTemplateOperator({'poly': {'iqu': poly}, 't2p': {'qu': leak}}, 3, 'IQU')
        >>> {name: sorted(legs) for name, legs in op.in_structure.items()}
        {'poly': ['i', 'q', 'u'], 't2p': ['q', 'u']}
    """

    bases: dict[str, dict[str, Basis]]
    stokes: ValidStokesLiteral = field(metadata={'static': True})

    def __post_init__(self) -> None:
        # `stokes` declares the leg axis: every template must be keyed by groups of its legs.
        for name, legged in self.bases.items():
            if isinstance(legged, Basis):
                msg = f'template {name!r} needs its bases keyed by Stokes leg of {self.stokes!r}'
                raise TypeError(msg)
            legs = ''.join(legged)
            if extra := sorted({leg for leg in legs if leg not in self.legs}):
                msg = (
                    f'template {name!r} has legs {extra} outside stokes={self.stokes!r} '
                    f'(expected {list(self.legs)})'
                )
                raise ValueError(msg)
            if repeated := sorted({leg for leg in legs if legs.count(leg) > 1}):
                raise ValueError(f'template {name!r} has legs {repeated} in several groups')
        super().__post_init__()

    @property
    def legs(self) -> tuple[StokesLeg, ...]:
        """The Stokes legs the TOD carries, as the amplitudes key them."""
        return cast(tuple[StokesLeg, ...], tuple(s.lower() for s in self.stokes))

    @property
    def bases_by_leg(self) -> dict[str, dict[StokesLeg, Basis]]:
        return {
            name: {cast(StokesLeg, leg): basis for group, basis in legged.items() for leg in group}
            for name, legged in self.bases.items()
        }

    def _amplitude_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.tree.map(self._amplitude_leaf, self.bases_by_leg, is_leaf=is_basis)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        stream = self._stream_structure()
        return Stokes.class_for(self.stokes).structure_for(stream.shape, stream.dtype)

    def _streams(self, tod: PyTree[Array]) -> dict[StokesLeg, Array]:
        return dict(zip(self.legs, tod.data, strict=True))

    def _bases_on(self, leg: StokesLeg) -> dict[str, Basis]:
        """The templates covering one leg, keyed by name."""
        return {name: on[leg] for name, on in self.bases_by_leg.items() if leg in on}

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        streams = []
        for leg in self.legs:
            bases = self._bases_on(leg)
            streams.append(self._expand_stream(bases, {name: x[name][leg] for name in bases}))
        return Stokes.class_for(self.stokes).from_array(jnp.stack(streams, axis=0))

    def project(self, tod: PyTree[Array]) -> PyTree[Array]:
        streams = self._streams(tod)
        return {
            name: {leg: basis._project_detectors(streams[leg]) for leg, basis in legged.items()}
            for name, legged in self.bases_by_leg.items()
        }


class _TemplateOperatorTranspose(TransposeOperator):
    operator: AbstractTemplateOperator

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        return self.operator.project(x)
