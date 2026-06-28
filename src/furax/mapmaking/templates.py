"""Template operators for fitting structured nuisance signals out of the data.

A template turns a small set of per-detector amplitudes into a time stream, so
the mapmaker can fit and remove unwanted but predictable signals (slow drifts,
scan-synchronous pickup, HWP-synchronous lines, T-to-P leakage).

The building block is a `Basis`: a small set of functions of time (Legendre
polynomials, HWP harmonics, ...). Going from amplitudes to a signal is `expand`;
the reverse is `project`. A `PerDetectorTemplate` then gives every detector its
own copy (or its own basis), so each detector fits its own amplitudes.

A few `Basis` flavours trade memory for structure:
- `TensorBasis`: stores every basis function value directly, as a dense array
  (optionally on a coarser time grid, `q > 1`, to trade resolution for memory).
- `KroneckerBasis`: a product of independent factors (e.g. azimuth x HWP), stored
  factored to save memory.
- `SegmentedBasis`: each sample belongs to one segment (e.g. one scan interval),
  stored sparsely instead of as a mostly-zero dense array.
- `WindowedBasis`: each sample reads a fixed window of overlapping blocks, the
  overlapping generalisation of `SegmentedBasis`.

Build several templates and combine them by wrapping each in a
`PerDetectorTemplate` and stacking with `BlockRowOperator`.
"""

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import field
from itertools import chain
from math import prod
from typing import Any, Self

import jax
import numpy as np
from jax import Array, ShapeDtypeStruct
from jax import numpy as jnp
from jax.scipy.linalg import cho_solve
from jaxtyping import DTypeLike, Float, Int, PyTree

from furax import AbstractLinearOperator, idempotent, symmetric
from furax.core import TransposeOperator
from furax.math import bspline, quaternion
from furax.obs import HWPOperator, LinearPolarizerOperator
from furax.obs.landscapes import HorizonLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import ValidStokesType

from .config import BinsConfig, PolynomialOrders


class Basis(AbstractLinearOperator):
    """A set of template functions of time used to model a structured signal.

    The basis functions `b_k` are each evaluated at the same `n_points` time samples.
    Conceptually, we can think of them as columns of a matrix `B`. The modelled signal
    is then a linear combination `s = B a`, i.e. `s(t) = Σ_k a_k b_k(t)` where each
    `a_k` is the amplitude of the template function `b_k`. The index `k` may be multi
    dimensional.

    Two main operations:

    - `expand` (synthesis, amplitudes to signal): `s = B(a)`.
    - `project` (analysis, signal to amplitudes): `a = B.T(s)`.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the basis index."""
        return self.in_structure.shape  # type: ignore[no-any-return]

    @property
    @abstractmethod
    def n_points(self) -> int:
        """Number of sample points at which basis functions are evaluated."""

    @property
    def size(self) -> int:
        """Total number of basis functions (product of `shape`)."""
        return prod(self.shape)

    @property
    def dtype(self) -> DTypeLike:
        return self.in_structure.dtype  # type: ignore[no-any-return]

    @property
    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.n_points,), self.dtype)

    @abstractmethod
    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' samp']:
        """Synthesize signal from coefficients."""

    @abstractmethod
    def project(self, signal: Float[Array, ' samp']) -> Float[Array, '*shape']:
        """Project signal onto basis."""

    def mv(self, x: Float[Array, '*shape']) -> Float[Array, ' samp']:
        return self.expand(x)

    def transpose(self) -> AbstractLinearOperator:
        return _BasisTranspose(self)


class _BasisTranspose(TransposeOperator):
    operator: Basis

    def mv(self, x: Float[Array, ' samp']) -> Float[Array, '*shape']:
        return self.operator.project(x)


class TensorBasis(Basis):
    """Dense basis: the matrix `B` is stored explicitly.

    Holds `values[k, t] = b_k(t)` as a single dense array, with no assumed structure.
    When `B` factorises over independent variables, `KroneckerBasis`
    represents the same map with less memory; when each sample belongs to one interval,
    `SegmentedBasis` avoids storing the mostly-zero per-interval blocks.

    With `q > 1` the values are stored on a `q`-times coarser time grid to save memory:
    synthesis (`expand`) holds each coarse value over its block of `q` samples and analysis
    (`project`) sums each block back down. The two are exact transposes. Hold/block-sum is a
    zeroth-order resampler, exact only for content well below the coarse-grid Nyquist
    frequency `sample_rate / 2q`, so choose `q` to keep that above the template's band edge.
    The default `q = 1` is the plain dense basis with no resampling.
    """

    values: Float[Array, '*shape samp_dec']
    q: int = field(metadata={'static': True}, default=1)
    n_full: int = field(metadata={'static': True}, default=0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.q < 1:
            raise ValueError(f'q must be >= 1, got {self.q}.')
        n_dec = self.values.shape[-1]
        # the coarse grid must be the q-block count covering n_full, i.e. n_dec = ceil(n_full / q).
        if not (n_dec - 1) * self.q < self.n_full <= n_dec * self.q:
            raise ValueError(f'n_dec={n_dec} inconsistent with n_full={self.n_full}, q={self.q}.')

    @classmethod
    def create(
        cls,
        values: Float[Array, '*shape samp_dec'],
        q: int = 1,
        n_full: int | None = None,
    ) -> Self:
        shape = values.shape[:-1]
        if n_full is None:
            # q == 1: the stored grid is already the full grid.
            n_full = values.shape[-1]
        return cls(
            values=values,
            q=q,
            n_full=n_full,
            in_structure=ShapeDtypeStruct(shape, values.dtype),
        )

    @property
    def n_points(self) -> int:
        return self.n_full

    @property
    def n_dec(self) -> int:
        return self.values.shape[-1]

    def _upsample(self, x: Float[Array, '... dec']) -> Float[Array, '... full']:
        # hold-interpolation: repeat each coarse sample q times, trim the padding tail.
        return jnp.repeat(x, self.q, axis=-1)[..., : self.n_full]

    def _downsample(self, s: Float[Array, '... full']) -> Float[Array, '... dec']:
        # adjoint of _upsample: sum each q-sample block. Pad the tail to a full block.
        pad = self.n_dec * self.q - self.n_full
        s = jnp.pad(s, [(0, 0)] * (s.ndim - 1) + [(0, pad)])
        return s.reshape(*s.shape[:-1], self.n_dec, self.q).sum(axis=-1)

    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' samp']:
        # integer axis labels. Index axes are 0..n-1, sample axis is n.
        n = len(self.shape)
        idx = tuple(range(n))
        out = jnp.einsum(coeffs, idx, self.values, (*idx, n), (n,))
        return out if self.q == 1 else self._upsample(out)

    def project(self, signal: Float[Array, ' samp']) -> Float[Array, '*shape']:
        n = len(self.shape)
        idx = tuple(range(n))
        if self.q != 1:
            signal = self._downsample(signal)
        return jnp.einsum(self.values, (*idx, n), signal, (n,), idx)


class KroneckerBasis(Basis):
    """Basis whose functions factorise over independent variables.

    Built from factor matrices `F_i` (e.g. an azimuth-polynomial set and an HWP-harmonic
    set), whose rows are the factor's functions of time. The basis function for
    multi-index `k = (k_0, ...)` is the elementwise product `b_k(t) = Π_i F_i[k_i, t]`.
    Equivalent to a `TensorBasis` holding the full outer product, but kept factored:
    memory scales as `Σ_i d_i` rather than `Π_i d_i` columns of length `n_points`.

    Use when the basis cleanly separates over independent variables.
    """

    factors: tuple[Float[Array, 'd samp'], ...]

    @classmethod
    def create(cls, factors: tuple[Float[Array, 'd samp'], ...]) -> Self:
        shape = tuple(f.shape[0] for f in factors)
        dtype = factors[0].dtype
        return cls(
            factors=factors,
            in_structure=ShapeDtypeStruct(shape, dtype),
        )

    @property
    def n_points(self) -> int:
        return self.factors[0].shape[-1]

    def _factor_operands(self) -> list[Any]:
        # interleaved einsum operands: factor i carries index axis i and the
        # shared sample axis n. e.g. F0,(0,n), F1,(1,n), ...
        n = len(self.shape)
        return list(chain.from_iterable((f, (i, n)) for i, f in enumerate(self.factors)))

    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' samp']:
        # explicit output (n,): sample axis n repeats across factors, name as output to keep it.
        n = len(self.shape)
        return jnp.einsum(coeffs, tuple(range(n)), *self._factor_operands(), (n,))

    def project(self, signal: Float[Array, ' samp']) -> Float[Array, '*shape']:
        # implicit output: repeated sample axis n is summed; index axes 0..n-1
        # each appear once and become the (sorted) output.
        n = len(self.shape)
        return jnp.einsum(*self._factor_operands(), signal, (n,))


class SegmentedBasis(Basis):
    """Basis partitioned into segments, each sample belonging to exactly one.

    The amplitude index is `(j, k)` = segment `j` × shared sub-basis function `k`, with
    `b_{j,k}(t) = [segment(t) = j] · v_k(t)`. Since every sample lies in a single
    segment, only that segment's amplitudes contribute to it.

    Stored sparsely as one segment id per sample plus one shared table `v_k(t)`, instead
    of the equivalent dense per-segment array (segments × functions × samples) that would
    be almost all zeros. The right choice when the segments form a partition (e.g. per-scan-interval
    Legendre polynomials); `KroneckerBasis` does not help here, as it assumes every
    factor is dense at every sample.

    Samples in no segment must have their `values` column pre-zeroed by the builder;
    their segment id is then irrelevant.
    """

    segment: Int[Array, ' samp']
    values: Float[Array, 'k samp']

    @classmethod
    def create(
        cls,
        segment: Int[Array, ' samp'],
        values: Float[Array, 'k samp'],
        n_segments: int,
    ) -> Self:
        k = values.shape[0]
        return cls(
            segment=segment,
            values=values,
            in_structure=ShapeDtypeStruct((n_segments, k), values.dtype),
        )

    @property
    def n_points(self) -> int:
        return self.values.shape[-1]

    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' samp']:
        # gather each sample's segment coefficients, then contract over the
        # sub-basis index against the shared per-sample values.
        picked = coeffs[self.segment]  # (n_points, k)
        return jnp.einsum('sk,ks->s', picked, self.values)

    def project(self, signal: Float[Array, ' samp']) -> Float[Array, '*shape']:
        # adjoint of expand: per-sample contribution scatter-added into its segment.
        contrib = self.values * signal[None, :]  # (k, n_points)
        zeros = jnp.zeros(self.shape, self.dtype)
        return zeros.at[self.segment].add(contrib.T)


class WindowedBasis(Basis):
    """Basis of overlapping blocks, each sample reading a fixed-width window of them.

    The amplitude index is a pair (block, sub-basis function). Every sample falls under a fixed
    number of consecutive blocks, weighting each by how far the sample sits inside it and each
    sub-basis function by its value there. The typical case is a cubic B-spline, where every
    sample lies under four consecutive knots.

    Stored sparsely as, per sample, the index of its first block, the window weights, and the
    shared sub-basis values, instead of the equivalent dense `TensorBasis` whose columns would
    be almost all zeros. The right choice when blocks overlap by a fixed amount;
    `SegmentedBasis` is the non-overlapping single-block-per-sample special case, kept separate
    to avoid storing its trivial unit window.

    The builder must keep every window inside the block range, pre-zeroing the weights of any
    sample whose window overhangs the ends.
    """

    offset: Int[Array, ' samp']
    block_weights: Float[Array, 'O samp']
    sub_values: Float[Array, 'k samp']

    @classmethod
    def create(
        cls,
        offset: Int[Array, ' samp'],
        block_weights: Float[Array, 'O samp'],
        sub_values: Float[Array, 'k samp'],
        n_blocks: int,
    ) -> Self:
        n_window, n_points = block_weights.shape
        k = sub_values.shape[0]
        if offset.shape != (n_points,) or sub_values.shape[1] != n_points:
            raise ValueError(
                f'sample axes disagree: offset {offset.shape}, block_weights {block_weights.shape}'
                f', sub_values {sub_values.shape}'
            )
        if n_window > n_blocks:
            raise ValueError(f'window ({n_window}) wider than block count ({n_blocks})')
        return cls(
            offset=offset,
            block_weights=block_weights,
            sub_values=sub_values,
            in_structure=ShapeDtypeStruct((n_blocks, k), sub_values.dtype),
        )

    @property
    def n_points(self) -> int:
        return self.offset.shape[0]

    def _block_indices(self) -> Int[Array, 'samp O']:
        # each sample's window: O consecutive block ids starting at its offset.
        n_window = self.block_weights.shape[0]
        return self.offset[:, None] + jnp.arange(n_window)

    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' samp']:
        # gather each sample's window of block coefficients, contract over the sub-basis
        # index against the shared values and over the window against its taper.
        gathered = coeffs[self._block_indices()]  # (samp, O, k)
        return jnp.einsum('soj,os,js->s', gathered, self.block_weights, self.sub_values)

    def project(self, signal: Float[Array, ' samp']) -> Float[Array, '*shape']:
        # adjoint of expand: per-sample rank-one contribution scatter-added into its window.
        contrib = jnp.einsum('os,js,s->soj', self.block_weights, self.sub_values, signal)
        zeros = jnp.zeros(self.shape, self.dtype)
        return zeros.at[self._block_indices()].add(contrib)


def _bin_weights(
    x: Float[Array, ' samp'],
    n_bins: int,
    interpolate: bool,
    smooth: bool,
    dtype: DTypeLike,
) -> Float[Array, '{n_bins} samp']:
    """Assign each sample to a bin of `x`.

    Splits the range of `x` into `n_bins` equal bins. Entry `[j, s]` is how much sample
    `s` belongs to bin `j`. With `interpolate=False` this is one-hot: 1 for the bin the
    sample falls in, 0 elsewhere. With `interpolate=True` a sample near a bin edge is
    shared with its neighbour (triangular weights, or sin² hats if `smooth`), each
    sample's weights summing to 1.

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
    # linearly to zero one bin-width (``delta``) away. Shape (n_bins, n_samps).
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
    """Legendre polynomials of orders `min_order..max_order` (inclusive), evaluated on
    `u` already rescaled to `[-1, 1]`."""
    legs = jax.scipy.special.lpmn_values(max_order, max_order, u, is_normalized=False)
    return legs[0, min_order:, :].astype(dtype)


def _legendre(
    x: Float[Array, ' samp'],
    min_order: int,
    max_order: int,
    dtype: DTypeLike,
) -> Float[Array, '{max_order-min_order+1} samp']:
    """Legendre polynomials of orders `min_order..max_order` (inclusive), evaluated on
    `x` rescaled to `[-1, 1]` over its global range."""
    u = -1.0 + 2.0 * (x - jnp.min(x)) / jnp.ptp(x)
    return _legendre_values(u, min_order, max_order, dtype)


def _harmonics(
    angles: Float[Array, ' samp'],
    harmonics: int | Sequence[int],
    dtype: DTypeLike,
    *,
    dc: bool,
) -> Float[Array, ' rows samp']:
    """Harmonic basis `sin(k·angle), cos(k·angle)` for each harmonic `k`, optionally
    prepended with a constant (DC) row when `dc` is set.

    `harmonics` is either an int `n` (the harmonics `1..n`) or an explicit sequence of
    harmonic orders.
    """
    h = jnp.arange(1, harmonics + 1) if isinstance(harmonics, int) else jnp.asarray(harmonics)
    sines = jnp.sin(h[:, None] * angles[None, :])
    cosines = jnp.cos(h[:, None] * angles[None, :])
    parts = ([jnp.ones((1, angles.size), dtype=dtype)] if dc else []) + [sines, cosines]
    return jnp.concatenate(parts, axis=0).astype(dtype)


class PerDetectorTemplate(AbstractLinearOperator):
    """Turn a single-detector `Basis` into a per-detector template operator.

    Each detector is an independent block fitting its own amplitudes. Two modes:

    - `shared_basis=True` (default): all detectors use the same basis. Used by templates
      whose basis depends only on shared quantities (azimuth, HWP angle): polynomial,
      scan- and HWP-synchronous.
    - `shared_basis=False`: each detector has its own basis. Used by the T2P leakage
      template, where detector `d`'s basis is its own temperature stream.
    """

    operator: AbstractLinearOperator
    shared_basis: bool = field(default=True, metadata={'static': True})

    @classmethod
    def from_basis(cls, basis: Basis, n_dets: int, *, shared: bool = True) -> Self:
        """Build the per-detector operator over `n_dets` detectors from a single `basis`.

        `shared=True` uses one basis for all detectors; `shared=False` expects a
        per-detector basis, one per detector (see the class docstring).
        """
        return cls(
            operator=basis,
            shared_basis=shared,
            in_structure=jax.ShapeDtypeStruct((n_dets, *basis.shape), basis.dtype),
        )

    @property
    def out_structure(self) -> jax.ShapeDtypeStruct:
        n_dets = self.in_structure.shape[0]
        out = self.operator.out_structure
        return jax.ShapeDtypeStruct((n_dets, *out.shape), out.dtype)

    def mv(self, x: Float[Array, ' det *shape']) -> Float[Array, 'det samp']:
        if self.shared_basis:
            # broadcast the shared operator across detectors.
            return jax.vmap(self.operator.mv)(x)  # type: ignore[no-any-return]
        # slice the basis values on the detector axis in lockstep with x
        return jax.vmap(lambda op, xi: op.mv(xi), in_axes=(0, 0))(self.operator, x)  # type: ignore[no-any-return]

    def transpose(self) -> AbstractLinearOperator:
        return PerDetectorTemplate(
            self.operator.T, shared_basis=self.shared_basis, in_structure=self.out_structure
        )

    @classmethod
    def scan_synchronous(
        cls,
        legendre: PolynomialOrders,
        azimuth: Float[Array, ' samp'],
        n_dets: int,
        dtype: DTypeLike,
    ) -> Self:
        """Scan-synchronous (azimuth-only) template on a global Legendre basis."""
        legs = _legendre(azimuth, legendre.min_order, legendre.max_order, dtype)
        return cls.from_basis(TensorBasis.create(legs), n_dets=n_dets)

    @classmethod
    def binaz_synchronous(
        cls,
        bins: BinsConfig,
        azimuth: Float[Array, ' samp'],
        n_dets: int,
        dtype: DTypeLike,
    ) -> Self:
        """Binned azimuth-synchronous template, no HWP coupling.

        One amplitude per azimuth bin per detector.
        """
        weights = _bin_weights(azimuth, bins.n_bins, bins.interpolate, bins.smooth, dtype)
        return cls.from_basis(TensorBasis.create(weights), n_dets=n_dets)

    @classmethod
    def hwp_synchronous(
        cls,
        n_harmonics: int,
        hwp_angles: Float[Array, ' samp'],
        n_dets: int,
        dtype: DTypeLike,
    ) -> Self:
        """HWP-synchronous template: harmonics of the HWP angle, `k = 1..n_harmonics`."""
        matrix = _harmonics(hwp_angles, n_harmonics, dtype, dc=False)
        return cls.from_basis(TensorBasis.create(matrix), n_dets=n_dets)

    @classmethod
    def azhwp_synchronous(
        cls,
        legendre: PolynomialOrders,
        n_harmonics: int,
        azimuth: Float[Array, ' samp'],
        hwp_angles: Float[Array, ' samp'],
        n_dets: int,
        dtype: DTypeLike,
        scan_mask: Float[Array, ' samp'] | None = None,
    ) -> Self:
        """Azimuth-Legendre × HWP-harmonic template (Kronecker product of the two).

        `scan_mask` optionally zeroes the azimuth leg on flagged samples (e.g. to fit
        separate amplitudes per scan direction).
        """
        poly = _legendre(azimuth, legendre.min_order, legendre.max_order, dtype)
        if scan_mask is not None:
            poly = scan_mask[None, :] * poly
        harm = _harmonics(hwp_angles, n_harmonics, dtype, dc=True)
        return cls.from_basis(KroneckerBasis.create((poly, harm)), n_dets=n_dets)

    @classmethod
    def binazhwp_synchronous(
        cls,
        bins: BinsConfig,
        n_harmonics: int,
        azimuth: Float[Array, ' samp'],
        hwp_angles: Float[Array, ' samp'],
        n_dets: int,
        dtype: DTypeLike,
    ) -> Self:
        """Azimuth-binned × HWP-harmonic template (azimuth is always binned)."""
        bin_basis = _bin_weights(azimuth, bins.n_bins, bins.interpolate, bins.smooth, dtype)
        harm = _harmonics(hwp_angles, n_harmonics, dtype, dc=True)
        return cls.from_basis(KroneckerBasis.create((bin_basis, harm)), n_dets=n_dets)

    @classmethod
    def polynomial(
        cls,
        max_poly_order: int,
        intervals: Float[Array, 'n_intervals 2'],
        times: Float[Array, ' samp'],
        n_dets: int,
        dtype: DTypeLike,
        valid_mask: Float[Array, ' samp'] | None = None,
    ) -> Self:
        """A polynomial drift template, one polynomial per scanning interval.

        Each sample belongs to one interval and is fitted with Legendre orders
        `0..max_poly_order` over that interval.

        Assumes `intervals` are sorted, non-overlapping `[start, end)` rows. Samples in
        gaps or past the last interval get a zero basis column. `valid_mask` optionally
        zeroes flagged samples (1 = keep, 0 = drop) so they neither carry template
        signal nor constrain the fitted amplitudes.
        """
        n_samps = times.size
        n_intervals = intervals.shape[0]
        starts = intervals[:, 0]
        ends = intervals[:, 1]

        s = jnp.arange(n_samps)
        # interval id per sample: last interval whose start <= s (intervals sorted),
        # clamped into range. Gaps/out-of-range are caught by ``in_range`` below.
        segment = jnp.clip(jnp.searchsorted(starts, s, side='right') - 1, 0, n_intervals - 1)
        seg_start = starts[segment]
        seg_end = ends[segment]
        in_range = (s >= seg_start) & (s < seg_end)

        t0 = times[seg_start]
        span = jnp.where(seg_end > seg_start + 1, times[seg_end - 1] - t0, 1.0)
        # rescale each sample to [-1, 1] within its own interval; out-of-range
        # samples sit at 0 and are zeroed by ``in_range`` below.
        u = jnp.where(in_range, -1.0 + 2.0 * (times - t0) / span, 0.0)
        legs = _legendre_values(u, 0, max_poly_order, dtype)  # (k, n_samps)
        legs = legs * in_range[None, :]
        if valid_mask is not None:
            legs = legs * valid_mask[None, :].astype(dtype)

        basis = SegmentedBasis.create(segment.astype(jnp.int32), legs, n_intervals)
        return cls.from_basis(basis, n_dets=n_dets)

    @classmethod
    def temperature(
        cls,
        temperature: Float[Array, 'det samp'],
        dtype: DTypeLike,
        fit_band: tuple[float, float] | None = None,
        sample_rate: Float[Array, ''] | float = 1.0,
        decimation_factor: int = 1,
    ) -> Self:
        """Temperature-to-polarization leakage template.

        Each detector's basis is just its own temperature stream, so fitting one
        amplitude per detector estimates how much temperature leaks into its
        polarization.

        `fit_band=(f0, f1)` restricts the temperature basis to that frequency band (Hz),
        so the leakage is both estimated and removed only there, keeping the template a
        clean linear operator.

        `decimation_factor=q` stores the basis on a `q`-times coarser grid to cut memory; the
        coarse-grid Nyquist frequency `sample_rate / 2q` must stay above `f1`. As a rule
        of thumb keep it at a few times `f1` (`q ≲ sample_rate / 6·f1`): the fractional
        error on the fitted amplitude grows like `(f1 / (sample_rate / 2q))²`.

        Assumes `temperature` is already deglitched/gap-filled upstream: a glitch left in
        it would smear across the band and bias the fitted amplitude.
        """
        t = temperature
        if fit_band is not None:
            f0, f1 = fit_band
            freqs = jnp.fft.rfftfreq(t.shape[-1], d=1.0 / sample_rate)
            band = (freqs > f0) & (freqs < f1)
            t = jnp.fft.irfft(jnp.fft.rfft(t, axis=-1) * band, n=t.shape[-1], axis=-1)
        n_dets, n_full = t.shape
        q = decimation_factor
        if q > 1:
            # Block-average onto a q-times coarser grid: pad the tail to a whole block,
            # reshape (..., n_dec, q) and mean. ``TensorBasis`` hold-upsamples back to
            # ``n_full`` in synthesis (band-limits above sample_rate / 2q).
            n_dec = -(-n_full // q)  # ceil
            pad = n_dec * q - n_full
            tp = jnp.pad(t, [(0, 0), (0, pad)])
            t_dec = tp.reshape(n_dets, n_dec, q).mean(axis=-1)
            values = t_dec[:, None, :].astype(dtype)  # (det, k=1, dec)
            # per-detector basis: values carry a leading det axis sliced by ``from_basis``,
            # so ``in_structure`` is the single-detector shape (k=1,).
            basis = TensorBasis(
                values=values, q=q, n_full=n_full, in_structure=ShapeDtypeStruct((1,), dtype)
            )
        else:
            values = t[:, None, :].astype(dtype)  # (det, k=1, samp)
            basis = TensorBasis(
                values=values, n_full=n_full, in_structure=ShapeDtypeStruct((1,), dtype)
            )
        return cls.from_basis(basis, n_dets=n_dets, shared=False)

    @classmethod
    def none(cls, n_dets: int, n_samps: int, dtype: DTypeLike) -> Self:
        """Empty template: no amplitudes, zero output. Used to leave a Stokes leg
        untouched in a per-leg block (e.g. the I leg of the T2P template, which acts
        on Q/U only)."""
        # k-axis is 0, so this is a zero-size array; n_samps only sizes the einsum output.
        values = jnp.zeros((n_dets, 0, n_samps), dtype)
        basis = TensorBasis(
            values=values, n_full=n_samps, in_structure=ShapeDtypeStruct((0,), dtype)
        )
        return cls.from_basis(basis, n_dets=n_dets, shared=False)

    @classmethod
    def bspline_hwpss(
        cls,
        times: Float[Array, ' samp'],
        hwp_angles: Float[Array, ' samp'],
        n_dets: int,
        n_knots: int,
        harmonics: int | Sequence[int] = (4,),
        dtype: DTypeLike = jnp.float32,
        scanning_mask: Float[Array, ' samp'] | None = None,
    ) -> Self:
        """Spline-based HWP synchronous template.

        A cubic B-spline models the slowly time-varying amplitude of the HWP-synchronous
        signal: knot `j` carries a `(sin kχ, cos kχ)` pair for each harmonic `k`, so the
        amplitudes have shape `(K, 2*n_harmonics)` with `K = n_knots + 2`.

        Args:
            times: Per-sample timestamps used to place the spline knots.
            hwp_angles: Per-sample HWP angle `χ` (radians).
            n_dets: Number of detectors; each fits its own amplitudes.
            n_knots: Number of interior spline knots (see `SplineHWPSSConfig.resolve_n_knots`).
            harmonics: HWP harmonics to model, either an int `n` (the harmonics `1..n`)
                or an explicit sequence of orders.
            dtype: Floating dtype of the basis values.
            scanning_mask: Optional (N,) mask restricting basis to valid scanning intervals.
                1 = valid scan, 0 = turnaround/gap. Basis functions are zeroed outside valid
                regions, ensuring the template fits only over scanned data.
        """
        offset, weights = bspline.spline_window(
            times, n_knots, scanning_mask=scanning_mask
        )  # weights (samp, 4)
        sub_values = _harmonics(hwp_angles, harmonics, dtype, dc=False).astype(dtype)
        basis = WindowedBasis.create(
            offset, weights.T.astype(dtype), sub_values, n_blocks=n_knots + 2
        )
        return cls.from_basis(basis, n_dets=n_dets, shared=True)


class GroundTemplateOperator(AbstractLinearOperator):
    """Operator for ground signal templates. The template amplitudes
    form a two-dimensional (elevation, azimuth) IQU map of the ground
    observed that is shared across detectors for the observation range.
    This class only contains a factory method.
    All argument angles should be provided in radians.
    """

    @classmethod
    def create(
        cls,
        azimuth_resolution: float,
        elevation_resolution: float,
        boresight_azimuth: Float[Array, ' samps'],
        boresight_elevation: Float[Array, ' samps'],
        boresight_rotation: Float[Array, ' samps'],
        detector_quaternions: Float[Array, 'dets 4'],
        hwp_angles: Float[Array, ' samps'],
        stokes: ValidStokesType,
        dtype: DTypeLike,
        landscape: HorizonLandscape | None = None,
        chunk_size: int = 0,
    ) -> AbstractLinearOperator:
        # Compute landscape if not provided
        if landscape is None:
            horizon_landscape: HorizonLandscape = cls.get_landscape(
                azimuth_resolution=azimuth_resolution,
                elevation_resolution=elevation_resolution,
                boresight_azimuth=boresight_azimuth,
                boresight_elevation=boresight_elevation,
                detector_quaternions=detector_quaternions,
                stokes=stokes,
                dtype=dtype,
            )
        else:
            horizon_landscape = landscape

        # Azimuth increases in an opposite way to longitude
        boresight_quaternions = quaternion.from_lonlat_angles(
            -boresight_azimuth, boresight_elevation, boresight_rotation
        )
        _, _, det_gamma = quaternion.to_xieta_angles(detector_quaternions)

        n_dets = detector_quaternions.shape[0]
        n_samps = boresight_azimuth.size

        pointing = PointingOperator.create(
            horizon_landscape,
            boresight_quaternions,
            detector_quaternions,
            chunk_size=chunk_size,
        )

        polarizer = LinearPolarizerOperator.create(
            shape=(n_dets, n_samps),
            dtype=dtype,
            stokes=stokes,
            angles=det_gamma[:, None].astype(dtype),
        )

        if stokes == 'I':
            return polarizer @ pointing

        hwp = HWPOperator.create(
            shape=(n_dets, n_samps), dtype=dtype, stokes=stokes, angles=hwp_angles.astype(dtype)
        )

        return polarizer @ hwp @ pointing

    @classmethod
    def get_landscape(
        cls,
        azimuth_resolution: float,
        elevation_resolution: float,
        boresight_azimuth: Float[Array, ' samps'],
        boresight_elevation: Float[Array, ' samps'],
        detector_quaternions: Float[Array, 'dets 4'],
        stokes: ValidStokesType,
        dtype: DTypeLike,
    ) -> HorizonLandscape:
        # First, set up a grid of (az, el) pairs
        n_grid = 10
        az_grid = jnp.linspace(jnp.min(boresight_azimuth), jnp.max(boresight_azimuth), n_grid)
        el_grid = jnp.linspace(jnp.min(boresight_elevation), jnp.max(boresight_elevation), n_grid)
        az_mesh, el_mesh = jnp.meshgrid(az_grid, el_grid, indexing='ij')
        qbore_mesh = quaternion.from_lonlat_angles(
            -az_mesh, el_mesh, jnp.zeros_like(az_mesh)
        )  # (ndet,N_GRID,N_GRID,4)
        qfull_mesh = quaternion.qmul(
            qbore_mesh[None, :, :, :], detector_quaternions[:, None, None, :]
        )
        det_az_mesh, det_el_mesh, _ = quaternion.to_lonlat_angles(qfull_mesh)
        det_az_mesh = -det_az_mesh

        # Azimuth angle is first restricted to to [0,2pi),
        # and unwrapped along the elevation grid, azimuth grid, and detector axes in order
        det_az_mesh = jnp.unwrap(
            jnp.unwrap(jnp.unwrap(det_az_mesh % (2 * jnp.pi), axis=2), axis=1), axis=0
        )

        # Allow small margins
        az_min = jnp.min(det_az_mesh) - 1e-4
        az_max = jnp.max(det_az_mesh) + 1e-4
        el_min = jnp.min(det_el_mesh) - 1e-4
        el_max = jnp.max(det_el_mesh) + 1e-4

        n_alt = int(np.ceil((el_max - el_min) / elevation_resolution))
        n_az = int(np.ceil((az_max - az_min) / azimuth_resolution))

        landscape = HorizonLandscape(
            shape=(n_az, n_alt),
            altitude_limits=(el_min, el_max),
            azimuth_limits=(az_min, az_max),
            stokes=stokes,
            dtype=dtype,
        )

        return landscape


@symmetric
@idempotent
class POMMEProjectionOperator(AbstractLinearOperator):
    """Projection operator for the POMME mapmaker.

    Subtracts the mean of blocks of length `tau` from each detector's timestream.
    This is a symmetric and idempotent operator: A = I - P, where P is the
    block-averaging operator.
    """

    tau: int = field(metadata={'static': True})

    def __init__(
        self,
        tau: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        object.__setattr__(self, 'tau', tau)
        object.__setattr__(self, 'in_structure', in_structure)

    def mv(self, x: Float[Array, 'det samp']) -> Float[Array, 'det samp']:
        n_det, n_samp = self.in_structure.shape
        n_int, n_rem = divmod(n_samp, self.tau)
        y = x[:, : n_int * self.tau].reshape(n_det, n_int, self.tau)
        y = y - jnp.mean(y, axis=-1, keepdims=True)
        y = y.reshape(n_det, n_int * self.tau)
        if n_rem == 0:
            return y
        return jnp.concatenate([y, x[:, -n_rem:]], axis=1)


@idempotent
@symmetric
class SplineHWPSSProjectionOperator(AbstractLinearOperator):
    """Sequential spline HWPSS deprojection.

    Implements:

        D3 = I - T (TᵀT)^(-1) Tᵀ

    with:

        T = D1 B

    where B is the spline HWPSS template and D1 is the POMME projector.
    """

    basis: WindowedBasis
    pomme: POMMEProjectionOperator
    chol: Array
    coeff_shape: tuple[int, ...] = field(metadata={'static': True})

    @classmethod
    def create(
        cls,
        template: PerDetectorTemplate,
        pomme: POMMEProjectionOperator,
    ) -> Self:

        if not isinstance(template.operator, WindowedBasis):
            raise TypeError('template.operator must be a WindowedBasis')

        if not template.shared_basis:
            raise ValueError('Spline HWPSS basis must be shared between detectors')

        basis = template.operator

        gram = cls._assemble_gram(
            basis,
            pomme,
        )

        # small regularisation for empty spline knots
        eps = 1e-10 * jnp.mean(jnp.diag(gram))
        gram = gram + eps * jnp.eye(
            gram.shape[0],
            dtype=gram.dtype,
        )

        chol = jnp.linalg.cholesky(gram)

        return cls(
            basis=basis,
            pomme=pomme,
            chol=chol,
            coeff_shape=basis.shape,
            in_structure=pomme.in_structure,
        )

    def mv(
        self,
        x: Float[Array, 'det samp'],
    ) -> Float[Array, 'det samp']:

        # T^T x = B^T D1 x
        rhs = jax.vmap(
            self.basis.project,
            in_axes=0,
        )(self.pomme(x))

        det = rhs.shape[0]

        rhs_flat = rhs.reshape(det, -1)

        # solve:
        #
        # (B^T D1 B) a = B^T D1 x
        #
        coeff_flat = jax.vmap(
            lambda r: cho_solve(
                (self.chol, True),
                r,
            )
        )(rhs_flat)

        coeff = coeff_flat.reshape(
            det,
            *self.coeff_shape,
        )

        # T a = D1 B a
        model = jax.vmap(
            self.basis.expand,
            in_axes=0,
        )(coeff)

        model = self.pomme(model)

        return x - model  # type: ignore[no-any-return]

    @staticmethod
    def _assemble_gram(
        basis: WindowedBasis,
        pomme: POMMEProjectionOperator,
    ) -> Array:
        """Compute:

        G = Bᵀ D1 B

        """

        n_blocks, n_harm = basis.shape
        n_coeff = n_blocks * n_harm

        n_samp = basis.n_points
        tau = pomme.tau

        O = basis.block_weights.shape[0]

        offset = basis.offset
        weights = basis.block_weights
        harmonics = basis.sub_values

        dtype = basis.dtype

        gram = jnp.zeros(
            (n_coeff, n_coeff),
            dtype=dtype,
        )

        harm_index = jnp.arange(n_harm)

        # --------------------------------------------------
        # B^T B
        # --------------------------------------------------

        def add_sample(
            s: int,
            g: Array,
        ) -> Array:

            blocks = offset[s] + jnp.arange(O)

            local = weights[:, s, None] * harmonics[:, s][None, :]

            local = local.reshape(-1)

            indices = (blocks[:, None] * n_harm + harm_index[None, :]).reshape(-1)

            return g.at[indices[:, None], indices[None, :]].add(local[:, None] * local[None, :])

        gram = jax.lax.fori_loop(
            0,
            n_samp,
            add_sample,
            gram,
        )

        # --------------------------------------------------
        # subtract B^T P B
        #
        # P averages blocks of length tau
        # --------------------------------------------------

        n_blocks_tau = n_samp // tau

        def subtract_block(
            b: int,
            g: Array,
        ) -> Array:

            start = b * tau

            block_sum = jnp.zeros(
                n_coeff,
                dtype=dtype,
            )

            def add_inside(
                i: int,
                v: Array,
            ) -> Array:

                s = start + i

                blocks = offset[s] + jnp.arange(O)

                local = (weights[:, s, None] * harmonics[:, s][None, :]).reshape(-1)

                indices = (blocks[:, None] * n_harm + harm_index[None, :]).reshape(-1)

                return v.at[indices].add(local)

            block_sum = jax.lax.fori_loop(
                0,
                tau,
                add_inside,
                block_sum,
            )

            return g - jnp.outer(block_sum, block_sum) / tau

        gram = jax.lax.fori_loop(
            0,
            n_blocks_tau,
            subtract_block,
            gram,
        )

        return gram  # type: ignore[no-any-return]
