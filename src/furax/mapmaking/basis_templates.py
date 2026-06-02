"""Refactored template operators with a Basis abstraction.

Design
------
A template operator is `T = I_{n_dets} ⊗ B` where `B: R^{*shape} -> R^{n_points}`
is a per-detector linear map ("basis"). The operator algebra (mv/transpose) lives
once in `PerDetectorTemplate`. Concrete template families differ only in how they
build their `Basis`.

Layering
--------
- `Basis(AbstractLinearOperator)`: per-detector map, with static `in_structure`
  carrying the basis-index shape and `out_structure` carrying ``n_points``. mv =
  `expand` (synthesis), transpose.mv = `project` (analysis). Static structure
  survives `jax.lax.scan` stacking of operators across observations: the basis
  array leaves gain a leading obs axis, but the structure metadata does not, so
  einsum subscripts derived from the static shape stay correct.
- `PerDetectorTemplate(AbstractLinearOperator)`: vmaps the basis over the
  detector axis.

Bases:
- `TensorBasis`    : dense, arbitrary-dim index
- `KroneckerBasis` : N factors, Khatri-Rao on sample axis (arbitrary N)

Combine heterogeneous templates by wrapping each in its own
``PerDetectorTemplate`` and composing via ``BlockRowOperator``.

Ground templates do NOT fit this hierarchy (pointing couples det+samp to a
shared sky map). They stay outside.
"""

from abc import abstractmethod
from dataclasses import field
from itertools import chain
from math import prod
from typing import Any, Self

import jax
from jax import Array, ShapeDtypeStruct
from jax import numpy as jnp
from jaxtyping import DTypeLike, Float, Int

from furax import AbstractLinearOperator
from furax.core import TransposeOperator

from .config import BinsConfig, LegendreOrders


class Basis(AbstractLinearOperator):
    """A finite basis of functions ``{b_k(t)}`` sampled at ``n_points`` points.

    The basis index ``k`` is multi-dimensional in general: ``in_structure.shape``
    gives its layout and ``size = prod(in_structure.shape)`` the total number of
    basis functions. ``out_structure`` is a 1D array of length ``n_points``.

    A signal in this basis is the linear combination
    ``signal(t) = sum_k coeffs[k] * b_k(t)``. The two operations are:

    - :meth:`expand` (=synthesis): coefficients → signal,
      ``signal[s] = sum_k coeffs[k] * b_k(t_s)``. Bound as ``mv``.
    - :meth:`project` (=analysis): signal → coefficients,
      ``coeffs[k] = sum_s signal[s] * b_k(t_s)``. Bound as transpose ``mv``.

    Basis arrays (``values``, ``factors``, ...) gain a leading obs axis under
    ``jax.lax.scan``-stacking, but ``in_structure`` and ``out_structure`` are
    static and stay shape-correct, so subscripts derived from them remain valid.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the basis index (static, from in_structure)."""
        return self.in_structure.shape  # type: ignore[no-any-return]

    @property
    @abstractmethod
    def n_points(self) -> int:
        """Number of sample points at which basis functions are evaluated."""

    @property
    def size(self) -> int:
        """Total number of basis functions (product of ``shape``)."""
        return prod(self.shape)

    @property
    def independent_ndim(self) -> int:
        """Number of leading basis-index axes that are mutually independent in the
        ``W``-metric Gram ``Bᵀ diag(w) B`` — i.e. block-diagonal, no coupling across them.

        Used by template marginalisation to invert the Gram block-wise. The default ``0``
        means every basis function may couple to every other (the whole index is one dense
        block per detector), correct for dense bases. :class:`SegmentedBasis` overrides it to
        ``1``: its leading segment axis is a one-hot partition, so distinct segments never
        share a sample and their Gram blocks are exactly decoupled."""
        return 0

    @property
    def dtype(self) -> DTypeLike:
        return self.in_structure.dtype  # type: ignore[no-any-return]

    @property
    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.n_points,), self.dtype)

    @abstractmethod
    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' points']:
        """Synthesize signal from coefficients."""

    @abstractmethod
    def project(self, signal: Float[Array, ' points']) -> Float[Array, '*shape']:
        """Project signal onto basis."""

    def mv(self, x: Float[Array, '*shape']) -> Float[Array, ' points']:
        return self.expand(x)

    def transpose(self) -> AbstractLinearOperator:
        return _BasisTranspose(self)


class _BasisTranspose(TransposeOperator):
    operator: Basis

    def mv(self, x: Float[Array, ' points']) -> Float[Array, '*shape']:
        return self.operator.project(x)


class TensorBasis(Basis):
    """Dense basis with arbitrary multi-dimensional index.

    Stores all basis function values in a single array ``values`` of shape
    ``(*shape, n_points)``: ``values[k_0, ..., k_{N-1}, s] = b_k(t_s)`` where
    ``k = (k_0, ..., k_{N-1})`` is the basis index. No factorization assumed —
    use :class:`KroneckerBasis` if the basis separates over independent
    variables (smaller memory).

    Common shapes:

    - ``shape = (k,)`` — flat 1D dense basis.
    - ``shape = (n_intervals, k)`` — per-interval dense basis, zero-padded
      outside each interval (polynomial-style).
    """

    values: Float[Array, '*shape points']

    @classmethod
    def create(cls, values: Float[Array, '*shape points']) -> Self:
        shape = values.shape[:-1]
        return cls(
            values=values,
            in_structure=ShapeDtypeStruct(shape, values.dtype),
        )

    @property
    def n_points(self) -> int:
        return self.values.shape[-1]

    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' points']:
        # integer axis labels. Index axes are 0..n-1, sample axis is n.
        n = len(self.shape)
        idx = tuple(range(n))
        return jnp.einsum(coeffs, idx, self.values, (*idx, n), (n,))

    def project(self, signal: Float[Array, ' points']) -> Float[Array, '*shape']:
        n = len(self.shape)
        idx = tuple(range(n))
        return jnp.einsum(self.values, (*idx, n), signal, (n,), idx)


class DecimatedTensorBasis(Basis):
    """Dense basis stored on a ``q``-times decimated sample grid.

    Identical interface to :class:`TensorBasis` but ``values`` hold one column per
    *decimated* sample (``n_dec = ceil(n_full / q)``) instead of one per full sample,
    cutting memory ~``q``x. Synthesis hold-upsamples the decimated signal back to the
    full grid (each coarse sample repeated ``q`` times); projection block-sums the full
    signal down to the coarse grid. The two are exact transposes, so the operator stays
    self-consistent. The basis functions are thus piecewise-constant over ``q``-sample
    blocks — i.e. band-limited above ``sample_rate / 2q``.

    **Validity bound.** Block-average analysis + zero-order-hold synthesis is a crude
    (non-band-limited) resampler: it is accurate only when the basis content lives well
    below the decimated Nyquist ``sample_rate / 2q``. The ZOH staircase error on a
    component at frequency ``f`` scales as ``~(pi f q / sample_rate)^2 / 6``, so e.g. for
    a T2P template band-passed to ``f1`` (see :meth:`PerDetectorTemplate.temperature`)
    the fitted coefficient drifts from the full-rate value by roughly that fraction.
    Empirically (band ``0.01-0.1 Hz``, ``fs = 200 Hz``): ~1e-6 at ``q = 10``, ~2e-4 at
    ``q = 100``, ~3e-3 at ``q = 400`` (decimated Nyquist ``0.25 Hz``, just above the
    band). Keep ``q`` such that ``sample_rate / 2q`` comfortably exceeds the band edge.
    A spectral (rfft-truncate) up/down pair would be exact for any ``q`` with the band
    inside Nyquist, at the cost of an FFT per synthesis; not used here since synthesis
    runs in the CG hot loop and the bound above is easily met in practice.

    ``values`` should already be block-averaged onto the coarse grid by the builder
    (see :meth:`PerDetectorTemplate.temperature`).
    """

    values: Float[Array, '*shape points_dec']
    q: int = field(metadata={'static': True})
    n_full: int = field(metadata={'static': True})

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

    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' points']:
        n = len(self.shape)
        idx = tuple(range(n))
        dec = jnp.einsum(coeffs, idx, self.values, (*idx, n), (n,))
        return self._upsample(dec)

    def project(self, signal: Float[Array, ' points']) -> Float[Array, '*shape']:
        n = len(self.shape)
        idx = tuple(range(n))
        dec = self._downsample(signal)
        return jnp.einsum(self.values, (*idx, n), dec, (n,), idx)


class KroneckerBasis(Basis):
    """Basis built as a pointwise product (Khatri-Rao) of N sub-bases.

    Given ``N`` factor matrices ``F_i`` of shape ``(d_i, n_points)``, the basis
    functions are indexed by a multi-index ``k = (k_0, ..., k_{N-1})`` with
    ``shape = (d_0, ..., d_{N-1})``, and

        b_k(t_s) = prod_i F_i[k_i, s].

    Equivalent to a ``TensorBasis`` whose ``values`` is the outer product of
    the factors along the basis-index axes, but stored factored: memory is
    ``sum_i d_i * n_points`` instead of ``prod_i d_i * n_points``.

    Use when the basis separates as a product over independent variables
    (e.g. azimuth-polynomial × HWP-harmonic, with the polynomial and harmonic
    templates each pre-evaluated on the sample grid).
    """

    factors: tuple[Float[Array, 'd points'], ...]

    @classmethod
    def create(cls, factors: tuple[Float[Array, 'd points'], ...]) -> Self:
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

    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' points']:
        # explicit output (n,): sample axis n repeats across factors, name as output to keep it.
        n = len(self.shape)
        return jnp.einsum(coeffs, tuple(range(n)), *self._factor_operands(), (n,))

    def project(self, signal: Float[Array, ' points']) -> Float[Array, '*shape']:
        # implicit output: repeated sample axis n is summed; index axes 0..n-1
        # each appear once and become the (sorted) output.
        n = len(self.shape)
        return jnp.einsum(*self._factor_operands(), signal, (n,))


class SegmentedBasis(Basis):
    """Partitioned basis: each sample belongs to exactly one segment.

    Index ``(j, m)`` = segment ``j`` × shared sub-basis function ``m``. Every
    sample ``s`` is assigned to one segment ``segment[s]``, so synthesis gathers
    that segment's coefficients and contracts with the per-sample basis values::

        signal[s] = sum_m coeffs[segment[s], m] * values[m, s].

    Stores ``segment`` (``n_points`` ints) + ``values`` (``k, n_points``) — i.e.
    ``O((k + 1) * n_points)`` — instead of the equivalent dense
    :class:`TensorBasis` which would be ``O(n_segments * k * n_points)``, almost
    all zeros (each segment's rows vanish outside its own samples).

    Use when the segment axis is a one-hot PARTITION (e.g. per-scan-interval
    Legendre polynomials). NOT the same as :class:`KroneckerBasis`, which assumes
    every factor is dense and every multi-index is active at every sample — a
    Kronecker indicator factor would store the partition densely, wasting the
    one-hot sparsity in exactly the dominant ``n_segments * n_points`` term.

    Samples outside every segment must have their ``values`` column pre-zeroed by
    the builder; their ``segment`` entry is then irrelevant (gathers a zero
    contribution either way).
    """

    segment: Int[Array, ' points']
    values: Float[Array, 'k points']

    @property
    def independent_ndim(self) -> int:
        # leading axis is the segment partition: one-hot, so segments don't couple.
        return 1

    @classmethod
    def create(
        cls,
        segment: Int[Array, ' points'],
        values: Float[Array, 'k points'],
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

    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' points']:
        # gather each sample's segment coefficients, then contract over the
        # sub-basis index against the shared per-sample values.
        picked = coeffs[self.segment]  # (n_points, k)
        return jnp.einsum('sk,ks->s', picked, self.values)

    def project(self, signal: Float[Array, ' points']) -> Float[Array, '*shape']:
        # adjoint of expand: per-sample contribution scatter-added into its segment.
        contrib = self.values * signal[None, :]  # (k, n_points)
        zeros = jnp.zeros(self.shape, self.dtype)
        return zeros.at[self.segment].add(contrib.T)


def _bins(
    x: Float[Array, ' samp'],
    n_bins: int,
    interpolate: bool,
    smooth: bool,
    dtype: DTypeLike,
) -> Float[Array, 'bin samp']:
    """Per-sample bin weights over ``x``, shape ``(n_bins, n_samps)``.

    Without interpolation, a hard one-hot assignment of each sample to its bin.
    With interpolation, triangular (or smoothed sin^2) weights spread each sample
    over neighbouring bin centres, normalised to sum to one per sample.
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
) -> Float[Array, 'order samp']:
    """Legendre polynomials of orders ``min_order..max_order`` (inclusive), evaluated
    on ``u`` *already* rescaled to ``[-1, 1]``. Shape ``(n_orders, n_samps)``."""
    legs = jax.scipy.special.lpmn_values(max_order, max_order, u, is_normalized=False)
    return legs[0, min_order:, :].astype(dtype)


def _legendre(
    x: Float[Array, ' samp'],
    min_order: int,
    max_order: int,
    dtype: DTypeLike,
) -> Float[Array, 'order samp']:
    """Legendre polynomials of orders ``min_order..max_order`` (inclusive),
    evaluated on ``x`` rescaled to ``[-1, 1]`` over its global range."""
    u = -1.0 + 2.0 * (x - jnp.min(x)) / jnp.ptp(x)
    return _legendre_values(u, min_order, max_order, dtype)


def _harmonics(
    angles: Float[Array, ' samp'],
    n_harmonics: int,
    dtype: DTypeLike,
    *,
    dc: bool,
) -> Float[Array, 'harm samp']:
    """Harmonic basis ``[sin(k.), cos(k.)]`` for ``k = 1..n_harmonics``, optionally
    prepended with a constant (DC) row. Shape ``(2*n_harmonics [+ 1], n_samps)``."""
    h = jnp.arange(1, n_harmonics + 1)
    sines = jnp.sin(h[:, None] * angles[None, :])
    cosines = jnp.cos(h[:, None] * angles[None, :])
    parts = ([jnp.ones((1, angles.size), dtype=dtype)] if dc else []) + [sines, cosines]
    return jnp.concatenate(parts, axis=0).astype(dtype)


class PerDetectorTemplate(AbstractLinearOperator):
    """Lift a single-detector ``Basis`` to a detector-blocked template operator.

    The structure is always ``I_{n_dets} ⊗ basis``: one independent block per
    detector, so each detector fits its *own* amplitudes. The lift is a ``vmap``
    over the detector axis. The basis *functions* may be either:

    - ``shared_basis=True`` (default): the basis operator is broadcast — every
      detector uses the same ``b_k(t)``. This is the ``vmap`` of a single closed-over
      operator. Used by polynomial, scan-/HWP-synchronous, ... templates whose basis
      depends only on shared quantities.
    - ``shared_basis=False``: the basis ``values`` carry a leading detector axis and
      are *sliced* in lockstep with the input — detector ``i``'s amplitudes meet
      detector ``i``'s own basis. The only difference is whether the operator argument
      is broadcast (``in_axes=None``) or mapped (``in_axes=0``); same ``mv`` code. Used
      by the T2P leakage template, where each detector's basis is its own temperature.

    ``shared_basis`` is static so it survives ``jax.lax.scan`` stacking over
    observations (which adds a further leading obs axis to every array leaf).
    """

    operator: AbstractLinearOperator
    shared_basis: bool = field(default=True, metadata={'static': True})

    @classmethod
    def from_basis(cls, basis: Basis, n_dets: int, *, shared_basis: bool = True) -> Self:
        return cls(
            operator=basis,
            shared_basis=shared_basis,
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
        # slice the operator's array leaves (basis values) on the detector axis
        # in lockstep with x: detector i ↔ its own basis.
        return jax.vmap(lambda op, xi: op.mv(xi), in_axes=(0, 0))(self.operator, x)  # type: ignore[no-any-return]

    def transpose(self) -> AbstractLinearOperator:
        # I ⊗ operator  =>  transpose is I ⊗ operator.T, the same vmap wrapper.
        return PerDetectorTemplate(
            self.operator.T, shared_basis=self.shared_basis, in_structure=self.out_structure
        )

    @classmethod
    def scan_synchronous(
        cls,
        legendre: LegendreOrders,
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
        bin_basis = _bins(azimuth, bins.n_bins, bins.interpolate, bins.smooth, dtype)
        return cls.from_basis(TensorBasis.create(bin_basis), n_dets=n_dets)

    @classmethod
    def hwp_synchronous(
        cls,
        n_harmonics: int,
        hwp_angles: Float[Array, ' samp'],
        n_dets: int,
        dtype: DTypeLike,
    ) -> Self:
        matrix = _harmonics(hwp_angles, n_harmonics, dtype, dc=False)
        return cls.from_basis(TensorBasis.create(matrix), n_dets=n_dets)

    @classmethod
    def azhwp_synchronous(
        cls,
        legendre: LegendreOrders,
        n_harmonics: int,
        azimuth: Float[Array, ' samp'],
        hwp_angles: Float[Array, ' samp'],
        n_dets: int,
        dtype: DTypeLike,
        scan_mask: Float[Array, ' samp'] | None = None,
    ) -> Self:
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
        bin_basis = _bins(azimuth, bins.n_bins, bins.interpolate, bins.smooth, dtype)
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
        """Polynomial template as a segmented (per-interval) basis.

        Each sample belongs to one scanning interval, rescaled to ``[-1, 1]`` over
        that interval and evaluated on Legendre orders ``0..max_poly_order``. Rather
        than materialise a dense ``(n_intervals, k, n_samps)`` array (``n_intervals``×
        redundant — every interval's rows are zero outside its own samples), store a
        per-sample segment id plus one shared ``(k, n_samps)`` Legendre evaluation in
        a :class:`SegmentedBasis`: ``O((k+1)*n_samps)`` instead of
        ``O(n_intervals*k*n_samps)``. Scan-stackable across observations.

        Assumes ``intervals`` are sorted, non-overlapping ``[start, end)`` rows
        (true for scanning intervals). Samples in gaps (turnarounds) or beyond the
        last interval get a zero basis column. ``valid_mask`` is an optional
        per-sample weight (1 = keep, 0 = drop) additionally zeroing flagged samples,
        so they neither carry template signal nor constrain the fitted coefficients.
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
        decimate: int = 1,
    ) -> Self:
        """T2P leakage as a fitted per-detector template.

        Each detector's basis is the single function ``b(t) = T_d(t)`` — its own
        temperature stream (the demodulated ``I`` leg) — so one amplitude per detector
        estimates the T→P leakage coefficient ``lambda[d]`` (synthesis ``lambda[d] * T_d``).
        Unlike the shared-basis families the basis differs per detector, so the
        ``values`` carry a leading detector axis and the lift uses ``shared_basis=False``.

        ``fit_band=(f0, f1)`` band-passes the temperature basis to ``f0 < f < f1`` (Hz,
        on the ``sample_rate`` grid). The leakage is then both *estimated and removed*
        within that band only — unlike the old deprojection, which estimated in-band but
        subtracted broadband — keeping it a clean linear (symmetric) template.

        ``decimate=q`` stores the basis on a ``q``-times coarser grid (block-average +
        ZOH synthesis) to cut memory. Accurate only while ``sample_rate / 2q`` stays
        comfortably above ``f1``; see :class:`DecimatedTensorBasis` for the error bound.

        Note: ``fit_band`` band-passes the *full* temperature stream, so glitches/outliers
        in ``temperature`` smear (sinc-ring) across the whole basis at in-band frequencies.
        Since the GLS mask is applied only to the fit residual (not the basis), a glitch
        flagged in the data but left in ``temperature`` biases the coefficient (asymmetric
        masking of ``b`` vs the data). Assumes the input is already deglitched/gap-filled
        upstream — true in the standard pipeline.
        """
        t = temperature
        if fit_band is not None:
            f0, f1 = fit_band
            freqs = jnp.fft.rfftfreq(t.shape[-1], d=1.0 / sample_rate)
            band = (freqs > f0) & (freqs < f1)
            t = jnp.fft.irfft(jnp.fft.rfft(t, axis=-1) * band, n=t.shape[-1], axis=-1)
        n_dets, n_full = t.shape
        if decimate > 1:
            # Block-average onto a q-times coarser grid: pad the tail to a whole block,
            # reshape (..., n_dec, q) and mean. ``DecimatedTensorBasis`` hold-upsamples
            # back to ``n_full`` in synthesis (band-limits above sample_rate / 2q).
            n_dec = -(-n_full // decimate)  # ceil
            pad = n_dec * decimate - n_full
            tp = jnp.pad(t, [(0, 0), (0, pad)])
            t_dec = tp.reshape(n_dets, n_dec, decimate).mean(axis=-1)
            values = t_dec[:, None, :].astype(dtype)  # (det, k=1, dec)
            basis: Basis = DecimatedTensorBasis(
                values=values,
                q=decimate,
                n_full=n_full,
                in_structure=ShapeDtypeStruct((1,), dtype),
            )
        else:
            values = t[:, None, :].astype(dtype)  # (det, k=1, samp)
            basis = TensorBasis(values=values, in_structure=ShapeDtypeStruct((1,), dtype))
        return cls.from_basis(basis, n_dets=n_dets, shared_basis=False)

    @classmethod
    def none(
        cls,
        n_dets: int,
        n_samps: int,
        dtype: DTypeLike,
    ) -> Self:
        """Empty template: no amplitudes (``k = 0``), zero output. Used to leave a
        Stokes leg untouched in a per-leg block (e.g. the ``I`` leg of the T2P
        template, which acts on ``Q``/``U`` only)."""
        values = jnp.zeros((n_dets, 0, n_samps), dtype)
        basis = TensorBasis(values=values, in_structure=ShapeDtypeStruct((0,), dtype))
        return cls.from_basis(basis, n_dets=n_dets, shared_basis=False)
