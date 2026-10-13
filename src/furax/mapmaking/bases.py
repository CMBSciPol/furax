"""Bases of template functions of time, and the sparse layouts that store them.

A [`Basis`][] is a small set of functions of time (Legendre polynomials, HWP harmonics, ...) used to
model a structured signal. Going from amplitudes to a signal is `expand`; the reverse is `project`.
The template operators of [`furax.mapmaking.templates`][] combine several bases into one operator
over a TOD.

A few [`Basis`][] flavours trade memory for structure:

- [`TensorBasis`][]: stores every basis function value directly, as a dense array
  (optionally on a coarser time grid, `q > 1`, to trade resolution for memory).
- [`KroneckerBasis`][]: a product of independent factors (e.g. azimuth x HWP), stored
  factored to save memory.
- [`SegmentedBasis`][]: each sample belongs to one segment (e.g. one scan interval),
  stored sparsely instead of as a mostly-zero dense array.
- [`OffsetBasis`][]: one offset per interval of a fixed number of samples, the regular case of
  [`SegmentedBasis`][], which stores nothing.
- [`WindowedBasis`][]: each sample reads a fixed window of overlapping blocks, the
  overlapping generalisation of [`SegmentedBasis`][].

A basis is normally shared by every detector. When the functions differ from detector to detector,
as for the T-to-P leakage template, [`Basis.per_detector_stack`][] stacks one basis per detector
into a single [`Basis`][] of any flavour.
"""

from abc import abstractmethod
from dataclasses import field, fields
from functools import partial
from itertools import chain
from math import prod
from typing import Any, NamedTuple, Self

import jax
import jax.numpy as jnp
from jax import Array, ShapeDtypeStruct
from jaxtyping import DTypeLike, Float, Int

from furax import AbstractLinearOperator
from furax.core import TransposeOperator

__all__ = [
    'NoStructuredView',
    'Basis',
    'TensorBasis',
    'KroneckerBasis',
    'SegmentedBasis',
    'OffsetBasis',
    'WindowedBasis',
]

# Full float32 precision for basis contraction to ensure adjointness of `project` and `expand`
_einsum = partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


class NoStructuredView(Exception):
    """Raised when a basis cannot expose a structured (column-support) view."""


class _BasisColumns(NamedTuple):
    """Column-support view of a basis, from which self- and cross-Grams are built in one pass.

    Examples:
        A dense basis is the degenerate case: one block holding every amplitude, and every sample
        in it with a unit tap, so `values` is the matrix itself.

        >>> basis = TensorBasis(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        >>> view = basis.support()
        >>> view.n_blocks, view.blocks.ravel().tolist(), view.taps.ravel().tolist()
        (1, [0, 0, 0], [1.0, 1.0, 1.0])
        >>> view.as_matrix()  # the basis matrix, in the same layout as basis.as_matrix()
        Array([[1., 4.],
               [2., 5.],
               [3., 6.]], dtype=float32)

        Overlapping blocks are what make the Gram block-banded. Four samples over five blocks, one
        sub-basis function, each sample reading a window of three; the third sample shares its
        window with the second, and each carries its own taps, as a B-spline's weights depend on
        where the sample falls inside its span.

        >>> taps = jnp.array([[.2, .1, .4, .3], [.5, .6, .4, .5], [.3, .3, .2, .2]])
        >>> taps.shape  # (window, samp), transposed into (samp, window) by the view
        (3, 4)
        >>> offset = jnp.array([0, 1, 1, 2])  # first block of each sample's window
        >>> basis = WindowedBasis(offset, taps, jnp.ones((1, 4)), n_blocks=5)
        >>> basis.support().blocks.tolist()  # the window of blocks each sample reads
        [[0, 1, 2], [1, 2, 3], [1, 2, 3], [2, 3, 4]]
        >>> basis.support().as_matrix()  # the band the window traces out
        Array([[0.2, 0.5, 0.3, 0. , 0. ],
               [0. , 0.1, 0.6, 0.3, 0. ],
               [0. , 0.4, 0.4, 0.2, 0. ],
               [0. , 0. , 0.3, 0.5, 0.2]], dtype=float32)
    """

    blocks: Int[Array, 'samp window']
    taps: Float[Array, 'samp window']
    values: Float[Array, 'k samp']
    n_blocks: int

    def as_matrix(self) -> Float[Array, 'samp block_k']:
        """The dense basis matrix this view encodes, in the basis's own `as_matrix()` layout."""
        n_points, window = self.blocks.shape
        k = self.values.shape[0]
        rows = jnp.arange(n_points)
        matrix = jnp.zeros((n_points, self.n_blocks, k), self.values.dtype)
        for slot in range(window):  # window width is static
            contrib = self.taps[:, slot][:, None] * self.values.T  # (samp, k)
            matrix = matrix.at[rows, self.blocks[:, slot]].add(contrib)
        return matrix.reshape(n_points, self.n_blocks * k)


class Basis(AbstractLinearOperator):
    """A set of template functions of time used to model a structured signal.

    The basis functions `b_k` are each evaluated at the same `n_points` time samples. Conceptually,
    we can think of them as columns of a matrix `B`. The modelled signal is then a linear
    combination `s = B a`, i.e. `s(t) = Σ_k a_k b_k(t)` where each `a_k` is the amplitude of the
    template function `b_k`. The index `k` may be multi dimensional.

    Two main operations:

    - `expand` (synthesis, amplitudes to signal): `s = B(a)`.
    - `project` (analysis, signal to amplitudes): `a = B.T(s)`.
    """

    per_detector: bool = field(metadata={'static': True}, default=False, kw_only=True)
    """Whether the stored arrays carry a leading detector axis."""

    @property
    @abstractmethod
    def _index_shape(self) -> tuple[int, ...]:
        """Single-detector amplitude index shape, read off the stored arrays."""

    @property
    @abstractmethod
    def _array_dtype(self) -> DTypeLike:
        """Dtype of the stored basis values."""

    def __post_init__(self) -> None:
        # `in_structure` is derived, never passed: it can then never disagree with the arrays.
        if self.in_structure is not None:
            raise ValueError('in_structure is derived from the basis arrays; do not pass it')
        # `per_detector_stack` marks its result after unflattening, which does not run this,
        # so reaching here with the marker set means a stack built by hand: `shape` would
        # describe one detector while the arrays hold many.
        if self.per_detector:
            raise ValueError('per_detector is set by per_detector_stack, not by the constructor')
        # Construction is the only caller, so `_index_shape` always sees one detector's arrays.
        object.__setattr__(
            self, 'in_structure', ShapeDtypeStruct(self._index_shape, self._array_dtype)
        )
        super().__post_init__()

    @classmethod
    def per_detector_stack(cls, **attributes: Any) -> Self:
        """One basis per detector, stacked into a single basis.

        Use this for a template whose functions differ from detector to detector, such as the
        temperature-to-polarization leakage template, whose basis is each detector's own temperature
        stream. Available on every flavour.

        Call it as you would the constructor, with every array argument carrying a leading detector
        axis. All detectors then share one index shape, dtype and static metadata, read off detector
        0, and differ only in their array values. The result keeps that single-detector metadata, so
        `shape` and `in_structure` still describe one detector's amplitudes, which is what the
        template operators map over.

        Args:
            attributes: The constructor arguments. Array arguments carry a leading detector
                axis; static ones (`q`, `n_segments`, ...) are passed as usual.

        Returns:
            A basis whose arrays carry a leading detector axis.

        Raises:
            ValueError: If an array argument is missing, or they disagree on the detector count.

        Examples:
            >>> temperature = jnp.arange(6.0).reshape(3, 2)  # 3 detectors, 2 samples
            >>> basis = TensorBasis.per_detector_stack(values=temperature[:, None])
            >>> basis.shape, basis.values.shape, basis.per_detector
            ((1,), (3, 1, 2), True)
        """
        array_names = [f.name for f in fields(cls) if not f.metadata.get('static', False)]
        if missing := [name for name in array_names if name not in attributes]:
            raise ValueError(f'{cls.__name__} needs a detector-stacked {", ".join(missing)}')

        stacked = [attributes[name] for name in array_names]
        if len(n_dets := {leaf.shape[0] for leaf in jax.tree.leaves(stacked)}) != 1:
            raise ValueError(f'detector axes disagree: {sorted(n_dets)}')

        # Detector 0 fixes the metadata; unflatten then carries it over to the stack untouched,
        # which is the point: the single-detector index shape must survive the stacking.
        first = lambda x: jax.tree.map(lambda leaf: leaf[0], x)
        proto = cls(**{k: first(v) if k in array_names else v for k, v in attributes.items()})
        basis: Self = jax.tree.unflatten(jax.tree.structure(proto), jax.tree.leaves(stacked))
        object.__setattr__(basis, 'per_detector', True)
        return basis

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the basis index, for a single detector."""
        return self.in_structure.shape

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
        return self.in_structure.dtype

    @property
    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.n_points,), self.dtype)

    @abstractmethod
    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' samp']:
        """Synthesize signal from coefficients."""

    @abstractmethod
    def project(self, signal: Float[Array, ' samp']) -> Float[Array, '*shape']:
        """Project signal onto basis."""

    def _expand_detectors(self, coeffs: Float[Array, 'det ...']) -> Float[Array, 'det samp']:
        """`expand` of every detector's amplitudes at once, detectors on the leading axis.

        A per-detector stack expands each detector on its own basis, a shared basis expands them
        all on the same one. Flavours override this where a joint pass is cheaper.
        """
        return jax.vmap(lambda op, a: op.expand(a), in_axes=self._detector_axes)(self, coeffs)

    def _project_detectors(self, signals: Float[Array, 'det samp']) -> Float[Array, 'det ...']:
        """`project` of every detector's stream at once, detectors on the leading axis.

        A per-detector stack projects each stream on its own basis, a shared basis projects them
        all on the same one. Flavours override this where a joint pass is cheaper.
        """
        return jax.vmap(lambda op, s: op.project(s), in_axes=self._detector_axes)(self, signals)

    @property
    def _detector_axes(self) -> tuple[int | None, int]:
        # map over the basis's own detector axis when it has one, broadcast it when shared
        return (0 if self.per_detector else None, 0)

    def _dense_values(self) -> Float[Array, 'k samp'] | None:
        """The basis matrix, `(size, n_points)`, where forming it is cheap; `None` otherwise.

        The template operators expand every basis that returns one in a single matrix product.
        """
        return None

    def mv(self, x: Float[Array, '*shape']) -> Float[Array, ' samp']:
        return self.expand(x)

    def transpose(self) -> AbstractLinearOperator:
        return _BasisTranspose(self)

    def gram(self, weights: Float[Array, ' samp']) -> Float[Array, 'n w1 k k']:
        """The weighted Gram `Bᵀ diag(weights) B`, keeping only the blocks that can be nonzero.

        Two amplitudes interact only where some sample sees both, so blocks far enough apart never
        do. `bands[j, d]` is the `k×k` block coupling block `j` to block `j + d`, for `d = 0..w`,
        with `d = 0` the diagonal block; the blocks below the diagonal are the transposes of these,
        by symmetry. Shape `(n_blocks, w + 1, k, k)`: blocks that never overlap give `w = 0`, and a
        dense basis is the single block `(1, 1, k, k)`. Consumed by
        [`BandedCholeskyOperator.from_bands`][furax.linalg.BandedCholeskyOperator.from_bands].

        Raises:
            NoStructuredView: If this basis has no such form. Flavours implement `_gram`.
        """
        if self.per_detector:
            raise NoStructuredView
        return self._gram(weights)

    @abstractmethod
    def _gram(self, weights: Float[Array, ' samp']) -> Float[Array, 'n w1 k k']:
        """Flavours with no such form raise [`NoStructuredView`][]."""

    @property
    @abstractmethod
    def _n_blocks(self) -> int:
        """How many blocks of time the amplitudes split into; 1 when every function spans the TOD."""

    def support(self) -> _BasisColumns:
        """Column-support view (`_BasisColumns`) used to build self- and cross-Grams in one pass.

        Consumed by [`cross_gram`][furax.mapmaking.gram.cross_gram].

        Raises:
            NoStructuredView: If this basis has no such view. Flavours implement `_support`.
        """
        if self.per_detector:
            raise NoStructuredView
        return self._support()

    @abstractmethod
    def _support(self) -> _BasisColumns:
        """Flavours with no such view raise [`NoStructuredView`][]."""


class _BasisTranspose(TransposeOperator):
    operator: Basis

    def mv(self, x: Float[Array, ' samp']) -> Float[Array, '*shape']:
        return self.operator.project(x)


class TensorBasis(Basis):
    """Dense basis: the matrix `B` is stored explicitly.

    Holds `values[k, t] = b_k(t)` as a single dense array, with no assumed structure. When `B`
    factorises over independent variables, `KroneckerBasis` represents the same map with less
    memory; when each sample belongs to one interval, `SegmentedBasis` avoids storing the
    mostly-zero per-interval blocks.

    With `q > 1` the values are stored on a `q`-times coarser time grid to save memory: synthesis
    (`expand`) holds each coarse value over its block of `q` samples and analysis (`project`) sums
    each block back down. The two are exact transposes. Hold/block-sum is a zeroth-order resampler,
    exact only for content well below the coarse-grid Nyquist frequency `sample_rate / 2q`, so
    choose `q` to keep that above the template's band edge. The default `q = 1` is the plain dense
    basis with no resampling.
    """

    values: Float[Array, '*shape samp_dec']
    q: int = field(metadata={'static': True}, default=1)
    n_full: int = field(metadata={'static': True}, default=0)

    @property
    def _index_shape(self) -> tuple[int, ...]:
        return self.values.shape[:-1]

    @property
    def _array_dtype(self) -> DTypeLike:
        return self.values.dtype

    def __post_init__(self) -> None:
        if self.n_full == 0:
            # q == 1: the stored grid is already the full grid.
            object.__setattr__(self, 'n_full', self.values.shape[-1])
        super().__post_init__()
        if self.q < 1:
            raise ValueError(f'q must be >= 1, got {self.q}.')
        n_dec = self.values.shape[-1]
        # the coarse grid must be the q-block count covering n_full, i.e. n_dec = ceil(n_full / q).
        if not (n_dec - 1) * self.q < self.n_full <= n_dec * self.q:
            raise ValueError(f'n_dec={n_dec} inconsistent with n_full={self.n_full}, q={self.q}.')

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
        out = _einsum(coeffs, idx, self.values, (*idx, n), (n,))
        return out if self.q == 1 else self._upsample(out)

    def project(self, signal: Float[Array, ' samp']) -> Float[Array, '*shape']:
        n = len(self.shape)
        idx = tuple(range(n))
        if self.q != 1:
            signal = self._downsample(signal)
        return _einsum(self.values, (*idx, n), signal, (n,), idx)

    def _dense_values(self) -> Float[Array, 'k samp'] | None:
        if self.q != 1 or self.per_detector:
            return None
        return self.values.reshape(self.size, self.n_points)

    def _gram(self, weights: Float[Array, ' samp']) -> Float[Array, '1 1 k k']:
        # Dense Gram over the (single) coupled amplitude axis: one block, no band (w=0).
        if self.q != 1:
            raise NoStructuredView('gram unsupported for decimated (q>1) TensorBasis')
        if len(self.shape) != 1:
            raise NoStructuredView('gram supports 1-D TensorBasis amplitude only')
        v = self.values  # (k, samp)
        return _einsum('as,s,bs->ab', v, weights, v)[None, None]

    @property
    def _n_blocks(self) -> int:
        return 1

    def _support(self) -> _BasisColumns:
        # One global block, every sample in it (unit tap).
        if self.q != 1 or len(self.shape) != 1:
            raise NoStructuredView('support view: 1-D undecimated TensorBasis only')
        samp = self.n_points
        return _BasisColumns(
            blocks=jnp.zeros((samp, 1), jnp.int32),
            taps=jnp.ones((samp, 1), self.dtype),
            values=self.values,
            n_blocks=self._n_blocks,
        )


class KroneckerBasis(Basis):
    """Basis whose functions factorise over independent variables.

    Built from factor matrices `F_i` (e.g. an azimuth-polynomial set and an HWP-harmonic set), whose
    rows are the factor's functions of time. The basis function for multi-index `k = (k_0, ...)` is
    the elementwise product `b_k(t) = Π_i F_i[k_i, t]`. Equivalent to a `TensorBasis` holding the
    full outer product, but stored factored: memory scales as `Σ_i d_i` rather than `Π_i d_i` columns
    of length `n_points`. A basis shared by every detector still forms the full product transiently
    when applied, as one product serves every detector.

    Use when the basis cleanly separates over independent variables.
    """

    factors: tuple[Float[Array, 'd samp'], ...]

    @property
    def _index_shape(self) -> tuple[int, ...]:
        return tuple(f.shape[0] for f in self.factors)

    @property
    def _array_dtype(self) -> DTypeLike:
        return self.factors[0].dtype

    @property
    def n_points(self) -> int:
        return self.factors[0].shape[-1]

    def _factor_operands(self) -> list[Any]:
        # interleaved einsum operands: factor i carries index axis i and the
        # shared sample axis n. e.g. F0,(0,n), F1,(1,n), ...
        n = len(self.shape)
        return list(chain.from_iterable((f, (i, n)) for i, f in enumerate(self.factors)))

    def _product_values(self) -> Float[Array, 'k samp']:
        """The product basis `V[k, t] = Π_i F_i[k_i, t]`, flattened to `(size, n_points)`."""
        n = len(self.shape)
        v = _einsum(*self._factor_operands(), (*range(n), n))
        return v.reshape(self.size, self.n_points)

    def _dense_values(self) -> Float[Array, 'k samp'] | None:
        return None if self.per_detector else self._product_values()

    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' samp']:
        if self.per_detector:
            # each detector's own product would cost `size` TODs
            n = len(self.shape)
            return _einsum(coeffs, tuple(range(n)), *self._factor_operands(), (n,))
        # one product for every detector: the factored contraction would form `(det, d_1, samp)`
        return _einsum('k,ks->s', coeffs.reshape(self.size), self._product_values())

    def project(self, signal: Float[Array, ' samp']) -> Float[Array, '*shape']:
        if self.per_detector:
            # implicit output: repeated sample axis n is summed; index axes 0..n-1
            # each appear once and become the (sorted) output.
            n = len(self.shape)
            return _einsum(*self._factor_operands(), signal, (n,))
        return _einsum('ks,s->k', self._product_values(), signal).reshape(self.shape)

    def _gram(self, weights: Float[Array, ' samp']) -> Float[Array, '1 1 k k']:
        # No block structure: the shared per-sample weight couples all factor indices, so the
        # Gram is dense over the flattened product index (one k×k block, k = prod(shape)).
        v = self._product_values()
        return _einsum('kt,t,lt->kl', v, weights, v)[None, None]

    @property
    def _n_blocks(self) -> int:
        return 1

    def _support(self) -> _BasisColumns:
        # One global block over the materialised product basis (flattened index k = prod(shape)).
        samp = self.n_points
        return _BasisColumns(
            blocks=jnp.zeros((samp, 1), jnp.int32),
            taps=jnp.ones((samp, 1), self.dtype),
            values=self._product_values(),
            n_blocks=self._n_blocks,
        )


class SegmentedBasis(Basis):
    """Basis partitioned into segments, each sample belonging to exactly one.

    The amplitude index is `(j, k)` = segment `j` × shared sub-basis function `k`, with `b_{j,k}(t)
    = [segment(t) = j] · v_k(t)`. Since every sample lies in a single segment, only that segment's
    amplitudes contribute to it.

    Stored sparsely as one segment id per sample plus one shared table `v_k(t)`, instead of the
    equivalent dense per-segment array (segments × functions × samples) that would be almost all
    zeros. The right choice when the segments form a partition (e.g. per-scan-interval Legendre
    polynomials); `KroneckerBasis` does not help here, as it assumes every factor is dense at every
    sample.

    Segment ids must be non-decreasing along the samples, as they are for time intervals: the
    projection relies on it and gives wrong results otherwise. Samples in no segment must have their
    `values` column pre-zeroed by the builder; their segment id must still keep the order.
    """

    segment: Int[Array, ' samp']
    values: Float[Array, 'k samp']
    n_segments: int = field(metadata={'static': True})

    @property
    def _index_shape(self) -> tuple[int, ...]:
        return (self.n_segments, self.values.shape[0])

    @property
    def _array_dtype(self) -> DTypeLike:
        return self.values.dtype

    @property
    def n_points(self) -> int:
        return self.values.shape[-1]

    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' samp']:
        # gather each sample's segment coefficients, then contract over the
        # sub-basis index against the shared per-sample values.
        picked = coeffs[self.segment]  # (n_points, k)
        return _einsum('sk,ks->s', picked, self.values)

    def project(self, signal: Float[Array, ' samp']) -> Float[Array, '*shape']:
        # adjoint of expand: per-sample contribution scatter-added into its segment.
        contrib = self.values * signal[None, :]  # (k, n_points)
        zeros = jnp.zeros(self.shape, self.dtype)
        return zeros.at[self.segment].add(contrib.T)

    def _expand_detectors(self, coeffs: Float[Array, 'det seg k']) -> Float[Array, 'det samp']:
        if self.per_detector:
            return super()._expand_detectors(coeffs)
        # a loop over `k`: mapping `expand` over detectors would gather `(samp, det, k)`
        stream = jnp.zeros((coeffs.shape[0], self.n_points), self.dtype)
        for k in range(self.values.shape[0]):  # sub-basis size is static
            stream = stream + coeffs[:, self.segment, k] * self.values[k][None, :]
        return stream

    def _project_detectors(self, signals: Float[Array, 'det samp']) -> Float[Array, 'det seg k']:
        if self.per_detector:
            return super()._project_detectors(signals)

        # The transpose stays in the loop body, where it fuses into the product: hoisted out, it
        # would copy the signals.
        def one_function(values: Float[Array, ' samp']) -> Float[Array, 'seg det']:
            return jax.ops.segment_sum(
                signals.T * values[:, None], self.segment, self.n_segments, indices_are_sorted=True
            )

        sums = jax.lax.map(one_function, self.values)  # (k, seg, det)
        return jnp.transpose(sums, (2, 1, 0))

    def _gram(self, weights: Float[Array, ' samp']) -> Float[Array, 'seg 1 k k']:
        # Each sample's rank-one w·vvᵀ binned into its segment: block-diagonal, so w = 0.
        vw = self.values * weights[None, :]  # (k, samp)

        def row(vw_a: Float[Array, ' samp']) -> Float[Array, 'seg k']:
            return jax.ops.segment_sum(
                (vw_a[None, :] * self.values).T,
                self.segment,
                self.n_segments,
                indices_are_sorted=True,
            )

        rows = jax.lax.map(row, vw)  # (k_a, seg, k_b)
        return jnp.moveaxis(rows, 0, 1)[:, None]

    @property
    def _n_blocks(self) -> int:
        return self.n_segments

    def _support(self) -> _BasisColumns:
        # One block per sample: its segment. Out-of-range samples were pre-zeroed in `values`.
        samp = self.segment.shape[0]
        return _BasisColumns(
            blocks=self.segment[:, None],
            taps=jnp.ones((samp, 1), self.dtype),
            values=self.values,
            n_blocks=self._n_blocks,
        )


class OffsetBasis(Basis):
    """One constant offset per interval of `tau` consecutive samples.

    The amplitude index is the interval. The samples past the last whole interval (the tail) belong
    to none. This is the Pomme template
    ([`PommeProjectionOperator`][furax.mapmaking.pomme.PommeProjectionOperator]).

    Nothing is stored. As the intervals are regular, `project` and `expand` reshape the samples into
    `(interval, tau)` rather than scattering or gathering by interval like the equivalent
    [`SegmentedBasis`][], and the Gram is each interval's weighted sample count.

    Examples:
        >>> basis = OffsetBasis(2, 5, jnp.float32)
        >>> basis.expand(jnp.array([1.0, 2.0]))
        Array([1., 1., 2., 2., 0.], dtype=float32)
        >>> basis.project(jnp.arange(5.0))
        Array([1., 5.], dtype=float32)
    """

    tau: int = field(metadata={'static': True})
    n_samples: int = field(metadata={'static': True})
    value_dtype: DTypeLike = field(metadata={'static': True})

    @property
    def _index_shape(self) -> tuple[int, ...]:
        return (self.n_intervals,)

    @property
    def _array_dtype(self) -> DTypeLike:
        return self.value_dtype

    @property
    def n_points(self) -> int:
        return self.n_samples

    @property
    def n_intervals(self) -> int:
        """Number of whole intervals."""
        return self.n_samples // self.tau

    def expand(self, coeffs: Float[Array, ' n']) -> Float[Array, ' samp']:
        return self._expand_detectors(coeffs)

    def project(self, signal: Float[Array, ' samp']) -> Float[Array, ' n']:
        return self._project_detectors(signal)

    def _expand_detectors(self, coeffs: Float[Array, '*det n']) -> Float[Array, '*det samp']:
        # any number of leading axes: each interval's offset broadcast over its samples
        *batch, n = coeffs.shape
        steps = jnp.broadcast_to(coeffs[..., None], (*batch, n, self.tau)).reshape(*batch, -1)
        tail = self.n_samples - n * self.tau
        return jnp.pad(steps, [(0, 0)] * len(batch) + [(0, tail)])

    def _project_detectors(self, signals: Float[Array, '*det samp']) -> Float[Array, '*det n']:
        # any number of leading axes: each interval's samples summed, the tail dropped
        end = self.n_intervals * self.tau
        intervals = signals[..., :end].reshape(*signals.shape[:-1], self.n_intervals, self.tau)
        return intervals.sum(axis=-1)

    def _gram(self, weights: Float[Array, ' samp']) -> Float[Array, 'n 1 1 1']:
        # the indicators never overlap: a diagonal Gram, each interval's weighted sample count
        return self.project(weights)[:, None, None, None]

    @property
    def _n_blocks(self) -> int:
        return self.n_intervals

    def _support(self) -> _BasisColumns:
        # one block per sample, its interval; the tail is in none, so its values are zero
        samples = jnp.arange(self.n_samples)
        last = max(self.n_intervals - 1, 0)
        return _BasisColumns(
            blocks=jnp.minimum(samples // self.tau, last).astype(jnp.int32)[:, None],
            taps=jnp.ones((self.n_samples, 1), self.dtype),
            values=(samples < self.n_intervals * self.tau)[None].astype(self.dtype),
            n_blocks=self._n_blocks,
        )


class WindowedBasis(Basis):
    """Basis of overlapping blocks, each sample reading a fixed-width window of them.

    The amplitude index is a pair (block, sub-basis function). Every sample falls under a fixed
    number of consecutive blocks, weighting each by how far the sample sits inside it and each
    sub-basis function by its value there. The typical case is a cubic B-spline, where every sample
    lies under four consecutive knots.

    Stored sparsely as, per sample, the index of its first block, the window weights, and the shared
    sub-basis values, instead of the equivalent dense `TensorBasis` whose columns would be almost
    all zeros. The right choice when blocks overlap by a fixed amount; `SegmentedBasis` is the
    non-overlapping single-block-per-sample special case, kept separate to avoid storing its trivial
    unit window.

    Offsets must be non-decreasing along the samples, as they are for blocks laid out in time: the
    projection relies on it and gives wrong results otherwise. The builder must keep every window
    inside the block range, pre-zeroing the weights of any sample whose window overhangs the ends.
    """

    offset: Int[Array, ' samp']
    block_weights: Float[Array, 'O samp']
    sub_values: Float[Array, 'k samp']
    n_blocks: int = field(metadata={'static': True})

    @property
    def _index_shape(self) -> tuple[int, ...]:
        return (self.n_blocks, self.sub_values.shape[0])

    @property
    def _array_dtype(self) -> DTypeLike:
        return self.sub_values.dtype

    def __post_init__(self) -> None:
        super().__post_init__()
        n_window, n_points = self.block_weights.shape
        if self.offset.shape != (n_points,) or self.sub_values.shape[1] != n_points:
            msg = (
                f'sample axes disagree: offset {self.offset.shape}, block_weights '
                f'{self.block_weights.shape}, sub_values {self.sub_values.shape}'
            )
            raise ValueError(msg)
        if n_window > self.n_blocks:
            raise ValueError(f'window ({n_window}) wider than block count ({self.n_blocks})')

    @property
    def n_points(self) -> int:
        return self.offset.shape[-1]

    def _block_indices(self) -> Int[Array, 'samp O']:
        # each sample's window: O consecutive block ids starting at its offset.
        n_window = self.block_weights.shape[0]
        return self.offset[:, None] + jnp.arange(n_window)

    def expand(self, coeffs: Float[Array, '*shape']) -> Float[Array, ' samp']:
        # gather each sample's window of block coefficients, contract over the sub-basis
        # index against the shared values and over the window against its taper.
        gathered = coeffs[self._block_indices()]  # (samp, O, k)
        return _einsum('soj,os,js->s', gathered, self.block_weights, self.sub_values)

    def project(self, signal: Float[Array, ' samp']) -> Float[Array, '*shape']:
        # adjoint of expand: per-sample rank-one contribution scatter-added into its window.
        contrib = _einsum('os,js,s->soj', self.block_weights, self.sub_values, signal)
        zeros = jnp.zeros(self.shape, self.dtype)
        return zeros.at[self._block_indices()].add(contrib)

    def _project_detectors(self, signals: Float[Array, 'det samp']) -> Float[Array, 'det ...']:
        if self.per_detector:
            return super()._project_detectors(signals)
        # one scatter per window slot keeps each slot's block ids sorted, as `offset` is
        sums = jnp.zeros((self.n_blocks, signals.shape[0], self.shape[1]), self.dtype)
        for o in range(self.block_weights.shape[0]):  # window width is static
            taps = self.block_weights[o][None, :] * self.sub_values  # (k, samp)
            contrib = signals.T[:, :, None] * taps.T[:, None, :]  # (samp, det, k)
            sums = sums + jax.ops.segment_sum(
                contrib, self.offset + o, self.n_blocks, indices_are_sorted=True
            )
        return jnp.moveaxis(sums, 1, 0)

    def _gram(self, weights: Float[Array, ' samp']) -> Float[Array, 'block band k k']:
        """Block-banded `Bᵀ diag(weights) B` in one pass.

        Overlapping windows couple block `j` only to `|j'-j| < O` (`O` = window width), so the Gram
        is block-banded of half-width `O-1`, dense `k×k` within the band. Returns the upper band
        `bands[j, d] = G[(j,·), (j+d,·)]` for `d = 0..O-1` (`d=0` the symmetric diagonal block; the
        lower band is `bands[j, d]ᵀ` by symmetry). Cost `O(n_samples · O² · k²)` — no full-TOD
        column probe.
        """
        n_window = self.block_weights.shape[0]
        n_blocks, k = self.shape
        # u[o, a, t] = block_weights[o, t] · sub_values[a, t]: sample t's value for window slot o,
        # sub-basis a. The block it lands in is offset[t] + o.
        u = self.block_weights[:, None, :] * self.sub_values[None, :, :]  # (O, k, samp)
        uw = u * weights[None, None, :]  # weight folded into one side
        bands = jnp.zeros((n_blocks, n_window, k, k), self.dtype)
        # d = j'-j is static; slot o (block j = offset+o) couples to slot o+d (block j+d).
        for d in range(n_window):
            for o in range(n_window - d):
                contrib = _einsum('at,bt->tab', uw[o], u[o + d])  # (samp, k, k)
                bands = bands.at[self.offset + o, d].add(contrib)
        return bands

    @property
    def _n_blocks(self) -> int:
        return self.n_blocks

    def _support(self) -> _BasisColumns:
        # Each sample reads a window of O overlapping blocks, tapered by block_weights.
        return _BasisColumns(
            blocks=self._block_indices(),  # (samp, O)
            taps=self.block_weights.T,  # (samp, O)
            values=self.sub_values,  # (k, samp)
            n_blocks=self._n_blocks,
        )
