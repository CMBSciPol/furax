"""Bases of template functions of time, and the sparse layouts that store them.

A [`Basis`][] is a small set of functions of time (Legendre polynomials, HWP harmonics, ...) used to
model a structured signal. As an operator, it maps amplitudes to a signal; its transpose maps a
signal back to amplitudes. A [`Template`][furax.mapmaking.templates.Template] sees a basis through
the detectors and streams of a TOD.

A few [`Basis`][] flavours trade memory for structure:

- [`TensorBasis`][]: stores every basis function value directly, as a dense array.
- [`KroneckerBasis`][]: a product of independent factors (e.g. azimuth x HWP), stored
  factored to save memory.
- [`SegmentedBasis`][]: each sample belongs to one segment (e.g. one scan interval),
  stored sparsely instead of as a mostly-zero dense array.
- [`OffsetBasis`][]: one offset per interval of a fixed number of samples, the regular case of
  [`SegmentedBasis`][], which stores nothing.
- [`WindowedBasis`][]: each sample reads a fixed window of overlapping blocks, the
  overlapping generalisation of [`SegmentedBasis`][].
- [`DecimatedBasis`][]: any of the above, stored on a coarser time grid to trade resolution for
  memory.

A basis is normally shared by every detector. When the functions differ from detector to detector,
as for the T-to-P leakage template, a basis of any flavour can instead hold a batch of bases, one
per detector (see `batch_shape` in [`Basis`][]).
"""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import field, fields
from functools import partial
from itertools import chain
from math import prod
from typing import Any, Self

import jax
import jax.numpy as jnp
from jax import Array, ShapeDtypeStruct
from jaxtyping import DTypeLike, Float, Int

from furax import AbstractLinearOperator
from furax.core import TransposeOperator

__all__ = [
    'Basis',
    'TensorBasis',
    'KroneckerBasis',
    'SegmentedBasis',
    'OffsetBasis',
    'WindowedBasis',
    'DecimatedBasis',
]

# Full float32 precision for basis contraction, so that a basis and its transpose stay adjoint
_einsum = partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


class Basis(AbstractLinearOperator):
    """A set of template functions of time used to model a structured signal.

    The basis functions `b_k` are each evaluated at the same `n_points` time samples. Conceptually,
    we can think of them as columns of a matrix `B`. The modelled signal is then a linear
    combination `s = B a`, i.e. `s(t) = Σ_k a_k b_k(t)` where each `a_k` is the amplitude of the
    template function `b_k`. The index `k` may be multi dimensional.

    As an operator, the basis synthesises a signal from amplitudes, `s = B(a)`, and its transpose
    analyses a signal back into amplitudes, `a = B.T(s)`. Both accept any number of leading batch
    axes, e.g. one set of amplitudes per detector, and apply the basis to each batch element.

    A basis can also hold a batch of bases of one flavour and shape, for functions that differ from
    detector to detector, such as the temperature-to-polarization leakage template, whose basis is
    each detector's own temperature stream. Pass `batch_shape` to the constructor, with every array
    argument carrying these leading axes. They then broadcast against the batch axes of the
    amplitudes and of the signal, as in NumPy: with `batch_shape=(n_dets,)`, basis `i` expands
    amplitudes `i`.

    Examples:
        >>> temperature = jnp.arange(6.0).reshape(3, 2)  # 3 detectors, 2 samples
        >>> basis = TensorBasis(temperature[:, None], batch_shape=(3,))
        >>> basis.shape, basis.in_structure.shape, basis.out_structure.shape
        ((1,), (3, 1), (3, 2))
        >>> basis(jnp.array([[1.0], [2.0], [0.0]]))
        Array([[0., 1.],
               [4., 6.],
               [0., 0.]], dtype=float32)
    """

    batch_shape: tuple[int, ...] = field(metadata={'static': True}, default=(), kw_only=True)
    """Leading axes of the stored arrays, one basis per index; empty for a single basis."""

    @property
    @abstractmethod
    def _array_dtype(self) -> DTypeLike:
        """Dtype of the stored basis values."""

    def __post_init__(self) -> None:
        # `in_structure` is derived, never passed: it can then never disagree with the arrays.
        if self.in_structure is not None:
            raise ValueError('in_structure is derived from the basis arrays; do not pass it')
        if self.batch_shape:
            self._check_batch_axes()
        structure = ShapeDtypeStruct((*self.batch_shape, *self.shape), self._array_dtype)
        object.__setattr__(self, 'in_structure', structure)
        super().__post_init__()

    def _check_batch_axes(self) -> None:
        shapes = [leaf.shape for leaf in jax.tree.leaves(self)]
        if not shapes:
            raise ValueError(f'{type(self).__name__} stores no array to batch')
        n = len(self.batch_shape)
        if any(shape[:n] != self.batch_shape for shape in shapes):
            msg = f'batch_shape={self.batch_shape} must lead every array, got shapes {shapes}'
            raise ValueError(msg)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of the amplitude index, of one basis of the batch."""

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
        # Read off static metadata rather than traced through `mv`, which would see the extra
        # leading axis of a basis stacked across observations.
        return jax.ShapeDtypeStruct((*self.batch_shape, self.n_points), self.dtype)

    def mv(self, x: Float[Array, '...']) -> Float[Array, '*batch samp']:
        return self._map_batch(lambda basis, a: basis._expand(a), x, len(self.shape))

    def transpose(self) -> AbstractLinearOperator:
        return _BasisTranspose(self)

    @abstractmethod
    def _expand(self, coeffs: Float[Array, '...']) -> Float[Array, '*batch samp']:
        """Synthesis `B a` for a single basis, over any leading batch axes of `coeffs`."""

    @abstractmethod
    def _project(self, signal: Float[Array, '*batch samp']) -> Float[Array, '...']:
        """Analysis `Bᵀ s` for a single basis, over any leading batch axes of `signal`."""

    def _map_batch(self, f: Callable[[Self, Array], Array], x: Array, core_ndim: int) -> Array:
        """`f(basis, x)` for each basis of the batch, `x` broadcast against `batch_shape`.

        `x` has `core_ndim` trailing axes that `f` consumes, after its batch axes. The output
        carries the broadcast batch axes in front of what `f` returns.
        """
        if not self.batch_shape:
            return f(self, x)
        x_batch = x.shape[: x.ndim - core_ndim]
        batch = jnp.broadcast_shapes(x_batch, self.batch_shape)
        x = jnp.broadcast_to(x, (*batch, *x.shape[len(x_batch) :]))
        # the basis's batch axes are the trailing ones of `batch`, mapped one at a time
        n_extra = len(batch) - len(self.batch_shape)
        g = lambda basis, y: f(basis._with_batch_shape(()), y)
        for _ in self.batch_shape:
            g = jax.vmap(g, in_axes=(0, n_extra), out_axes=n_extra)
        return g(self, x)

    def _with_batch_shape(self, batch_shape: tuple[int, ...]) -> Self:
        """This basis, rebuilt on arrays that carry only `batch_shape` leading axes.

        `vmap` and `lax.map` pass the arrays that way, once they have sliced the other axes off.
        """
        excluded = {'in_structure', 'batch_shape'}
        names = [f.name for f in fields(self) if f.init and f.name not in excluded]
        return type(self)(**{name: getattr(self, name) for name in names}, batch_shape=batch_shape)

    def _dense_values(self) -> Float[Array, 'k samp'] | None:
        """The basis matrix, `(size, n_points)`, where forming it is cheap; `None` otherwise.

        The template operators expand every basis that returns one in a single matrix product.
        """
        return None

    def gram(self, weights: Float[Array, '*batch samp']) -> Float[Array, '*batch n w1 k k']:
        """The weighted Gram `Bᵀ diag(weights) B`, keeping only the blocks that can be nonzero.

        Two amplitudes interact only where some sample sees both, so blocks far enough apart never
        do. `bands[j, d]` is the `k×k` block coupling block `j` to block `j + d`, for `d = 0..w`,
        with `d = 0` the diagonal block; the blocks below the diagonal are the transposes of these,
        by symmetry. Shape `(n_blocks, w + 1, k, k)`: blocks that never overlap give `w = 0`, and a
        dense basis is the single block `(1, 1, k, k)`. Consumed by
        [`BandedCholeskyOperator.from_bands`][furax.linalg.BandedCholeskyOperator.from_bands].

        For a batch of bases, `weights` broadcasts against `batch_shape` and the bands carry the
        batch axes, one Gram per basis.
        """
        return self._map_batch(lambda basis, w: basis._gram(w), weights, 1)

    def _gram(self, weights: Float[Array, ' samp']) -> Float[Array, 'n w1 k k']:
        """`gram` for a single basis: that of its support, unless a flavour has a cheaper one."""
        return self.support()._gram(weights)

    @property
    @abstractmethod
    def _n_blocks(self) -> int:
        """How many blocks of time the amplitudes split into; 1 when every function spans the TOD."""

    @property
    def _support_window(self) -> int:
        """How many consecutive blocks each sample reads."""

        def taps(basis: Basis) -> Array:
            return basis._map_batch(lambda b, _: b.support().taps, jnp.zeros(basis.batch_shape), 0)

        return jax.eval_shape(taps, self).shape[-1]

    def _support_reach(self, bandwidth: int) -> Int[Array, '']:
        """How many blocks apart two blocks are that samples at most `bandwidth` apart read.

        Only samples where some function is nonzero count. The largest over a batch of bases.
        Assumes blocks are non-decreasing in time.
        """

        def single(basis: Basis, _: Array) -> Array:
            support = basis.support()
            n = support.n_points
            active = jnp.any(support.values != 0, axis=0)
            t = jnp.arange(n)
            last = jax.lax.cummax(jnp.where(active, t, -1))  # the last active sample up to each
            ahead = last[jnp.minimum(t + bandwidth, n - 1)]
            span = support.offset[ahead] - support.offset + support.taps.shape[-1] - 1
            return jnp.max(jnp.where(active, span, 0))

        return jnp.max(self._map_batch(single, jnp.zeros(self.batch_shape), 0))

    @abstractmethod
    def support(self) -> 'WindowedBasis':
        """This basis written as an equivalent [`WindowedBasis`][]: its support.

        Every flavour splits its amplitudes into blocks of sub-basis functions, each sample reading
        a window of consecutive blocks. Written in that common form, any two bases give their
        self- and cross-Grams in one pass. Defined for a single basis, with an empty
        `batch_shape`.

        Examples:
            A dense basis is the degenerate case: one block holding every amplitude, and every
            sample in it with a unit tap, so `values` is the matrix itself.

            >>> basis = TensorBasis(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            >>> support = basis.support()
            >>> support.n_blocks, support.offset.tolist(), support.taps.ravel().tolist()
            (1, [0, 0, 0], [1.0, 1.0, 1.0])
            >>> support.values
            Array([[1., 2., 3.],
                   [4., 5., 6.]], dtype=float32)
        """


class _BasisTranspose(TransposeOperator):
    operator: Basis

    def mv(self, x: Float[Array, '*batch samp']) -> Float[Array, '...']:
        return self.operator._map_batch(lambda basis, s: basis._project(s), x, 1)


class TensorBasis(Basis):
    """Dense basis: the matrix `B` is stored explicitly.

    Holds `values[k, t] = b_k(t)` as a single dense array, with no assumed structure. When `B`
    factorises over independent variables, `KroneckerBasis` represents the same map with less
    memory; when each sample belongs to one interval, `SegmentedBasis` avoids storing the
    mostly-zero per-interval blocks. To store the values on a coarser time grid, wrap the basis in
    a [`DecimatedBasis`][].
    """

    values: Float[Array, '*shape samp']

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape[len(self.batch_shape) : -1]

    @property
    def _array_dtype(self) -> DTypeLike:
        return self.values.dtype

    @property
    def n_points(self) -> int:
        return self.values.shape[-1]

    def _flat_values(self) -> Float[Array, 'k samp']:
        return self.values.reshape(self.size, self.n_points)

    def _expand(self, coeffs: Float[Array, '...']) -> Float[Array, '*batch samp']:
        batch = coeffs.shape[: coeffs.ndim - len(self.shape)]
        flat = coeffs.reshape(*batch, self.size)
        return _einsum('...k,ks->...s', flat, self._flat_values())

    def _project(self, signal: Float[Array, '*batch samp']) -> Float[Array, '...']:
        out = _einsum('...s,ks->...k', signal, self._flat_values())
        return out.reshape(*signal.shape[:-1], *self.shape)

    def _dense_values(self) -> Float[Array, 'k samp'] | None:
        return None if self.batch_shape else self._flat_values()

    def _gram(self, weights: Float[Array, ' samp']) -> Float[Array, '1 1 k k']:
        # Dense Gram over the flattened amplitude index: one block, no band (w=0).
        v = self._flat_values()
        return _einsum('as,s,bs->ab', v, weights, v)[None, None]

    @property
    def _n_blocks(self) -> int:
        return 1

    def support(self) -> 'WindowedBasis':
        # One global block over the flattened amplitude index, every sample in it (unit tap).
        return _single_block(self._flat_values())


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
    def shape(self) -> tuple[int, ...]:
        return tuple(f.shape[-2] for f in self.factors)

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
        return None if self.batch_shape else self._product_values()

    def _expand(self, coeffs: Float[Array, '...']) -> Float[Array, '*batch samp']:
        n = len(self.shape)
        batch = coeffs.shape[: coeffs.ndim - n]
        if not batch:
            # a single amplitude set, as each basis of a batch gets: the factored contraction
            # never forms the `size`-TOD product
            return _einsum(coeffs, tuple(range(n)), *self._factor_operands(), (n,))
        # one product for every batch element: the factored contraction would form
        # `(*batch, d_1, samp)`
        flat = coeffs.reshape(*batch, self.size)
        return _einsum('...k,ks->...s', flat, self._product_values())

    def _project(self, signal: Float[Array, '*batch samp']) -> Float[Array, '...']:
        if signal.ndim == 1:
            # implicit output: repeated sample axis n is summed; index axes 0..n-1
            # each appear once and become the (sorted) output.
            n = len(self.shape)
            return _einsum(*self._factor_operands(), signal, (n,))
        out = _einsum('...s,ks->...k', signal, self._product_values())
        return out.reshape(*signal.shape[:-1], *self.shape)

    def _gram(self, weights: Float[Array, ' samp']) -> Float[Array, '1 1 k k']:
        # No block structure: the shared per-sample weight couples all factor indices, so the
        # Gram is dense over the flattened product index (one k×k block, k = prod(shape)).
        v = self._product_values()
        return _einsum('kt,t,lt->kl', v, weights, v)[None, None]

    @property
    def _n_blocks(self) -> int:
        return 1

    def support(self) -> 'WindowedBasis':
        # One global block over the materialised product basis (flattened index k = prod(shape)).
        return _single_block(self._product_values())


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
    def shape(self) -> tuple[int, ...]:
        return (self.n_segments, self.values.shape[-2])

    @property
    def _array_dtype(self) -> DTypeLike:
        return self.values.dtype

    @property
    def n_points(self) -> int:
        return self.values.shape[-1]

    def _expand(self, coeffs: Float[Array, '*batch seg k']) -> Float[Array, '*batch samp']:
        # a loop over `k`: gathering whole segment rows would form `(*batch, samp, k)`
        stream = jnp.zeros((*coeffs.shape[:-2], self.n_points), self.dtype)
        for k in range(self.values.shape[0]):  # sub-basis size is static
            stream = stream + coeffs[..., self.segment, k] * self.values[k]
        return stream

    def _project(self, signal: Float[Array, '*batch samp']) -> Float[Array, '*batch seg k']:
        batch = signal.shape[:-1]
        signals = signal.reshape(-1, self.n_points)

        # The transpose stays in the loop body, where it fuses into the product: hoisted out, it
        # would copy the signals.
        def one_function(values: Float[Array, ' samp']) -> Float[Array, 'seg batch']:
            return jax.ops.segment_sum(
                signals.T * values[:, None], self.segment, self.n_segments, indices_are_sorted=True
            )

        sums = jax.lax.map(one_function, self.values)  # (k, seg, batch)
        return jnp.transpose(sums, (2, 1, 0)).reshape(*batch, *self.shape)

    @property
    def _n_blocks(self) -> int:
        return self.n_segments

    def support(self) -> 'WindowedBasis':
        # One block per sample: its segment. Out-of-range samples were pre-zeroed in `values`.
        taps = jnp.ones((self.n_points, 1), self.dtype)
        return WindowedBasis(self.segment, taps, self.values, self.n_segments)


class OffsetBasis(Basis):
    """One constant offset per interval of `tau` consecutive samples.

    The amplitude index is the interval. The samples past the last whole interval (the tail) belong
    to none. This is the Pomme template
    ([`PommeProjectionOperator`][furax.mapmaking.pomme.PommeProjectionOperator]).

    Nothing is stored. As the intervals are regular, the basis and its transpose reshape the samples
    into `(interval, tau)` rather than gathering or scattering by interval like the equivalent
    [`SegmentedBasis`][], and the Gram is each interval's weighted sample count.

    Examples:
        >>> basis = OffsetBasis(2, 5, jnp.float32)
        >>> basis(jnp.array([1.0, 2.0]))
        Array([1., 1., 2., 2., 0.], dtype=float32)
        >>> basis.T(jnp.arange(5.0))
        Array([1., 5.], dtype=float32)
    """

    tau: int = field(metadata={'static': True})
    n_samples: int = field(metadata={'static': True})
    value_dtype: DTypeLike = field(metadata={'static': True})

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n_samples // self.tau,)  # whole intervals only

    @property
    def _array_dtype(self) -> DTypeLike:
        return self.value_dtype

    @property
    def n_points(self) -> int:
        return self.n_samples

    def _expand(self, coeffs: Float[Array, '*batch n']) -> Float[Array, '*batch samp']:
        # each interval's offset broadcast over its samples
        *batch, n = coeffs.shape
        steps = jnp.broadcast_to(coeffs[..., None], (*batch, n, self.tau)).reshape(*batch, -1)
        tail = self.n_samples - n * self.tau
        return jnp.pad(steps, [(0, 0)] * len(batch) + [(0, tail)])

    def _project(self, signal: Float[Array, '*batch samp']) -> Float[Array, '*batch n']:
        # each interval's samples summed, the tail dropped
        (n,) = self.shape
        intervals = signal[..., : n * self.tau].reshape(*signal.shape[:-1], n, self.tau)
        return intervals.sum(axis=-1)

    def _gram(self, weights: Float[Array, ' samp']) -> Float[Array, 'n 1 1 1']:
        # the indicators never overlap: a diagonal Gram, each interval's weighted sample count
        return self._project(weights)[:, None, None, None]

    @property
    def _n_blocks(self) -> int:
        return self.shape[0]

    def support(self) -> 'WindowedBasis':
        # one block per sample, its interval; the tail is in none, so its values are zero
        (n,) = self.shape
        samples = jnp.arange(self.n_samples)
        offset = jnp.minimum(samples // self.tau, max(n - 1, 0)).astype(jnp.int32)
        taps = jnp.ones((self.n_samples, 1), self.dtype)
        values = (samples < n * self.tau)[None].astype(self.dtype)
        return WindowedBasis(offset, taps, values, n)


class WindowedBasis(Basis):
    """Basis of overlapping blocks, each sample reading a fixed-width window of them.

    The amplitude index is a pair (block, sub-basis function). Sample `t` reads a window of
    consecutive blocks, the first being `offset[t]`, and its entry for the amplitude
    `(offset[t] + o, a)` is `taps[t, o] · values[a, t]`: each block weighted by where the sample
    sits inside it, each sub-basis function by its value there. The typical case is a cubic
    B-spline, where every sample lies under four consecutive knots.

    Stored sparsely, instead of as the equivalent dense `TensorBasis` whose columns would be almost
    all zeros. Every other flavour is a special case: a [`SegmentedBasis`][] has a window of one
    block, a [`TensorBasis`][] a single block. [`Basis.support`][] rewrites any basis in this form,
    from which its Grams are built.

    Offsets must be non-decreasing along the samples, as they are for blocks laid out in time: the
    transpose and the Gram rely on it and give wrong results otherwise. The builder must keep every
    window inside the block range, pre-zeroing the taps of any sample whose window overhangs the
    ends.

    Examples:
        Overlapping blocks are what make the Gram block-banded. Four samples over five blocks, one
        sub-basis function, each sample reading a window of three; the third sample shares its
        window with the second, and each carries its own taps, as a B-spline's weights depend on
        where the sample falls inside its span.

        >>> offset = jnp.array([0, 1, 1, 2])
        >>> taps = jnp.array([[.2, .5, .3], [.1, .6, .3], [.4, .4, .2], [.3, .5, .2]])
        >>> basis = WindowedBasis(offset, taps, jnp.ones((1, 4)), n_blocks=5)
        >>> basis.blocks.tolist()
        [[0, 1, 2], [1, 2, 3], [1, 2, 3], [2, 3, 4]]
        >>> basis.as_matrix()  # the band the windows trace out
        Array([[0.2, 0.5, 0.3, 0. , 0. ],
               [0. , 0.1, 0.6, 0.3, 0. ],
               [0. , 0.4, 0.4, 0.2, 0. ],
               [0. , 0. , 0.3, 0.5, 0.2]], dtype=float32)
    """

    offset: Int[Array, ' samp']
    """The first block of each sample's window."""
    taps: Float[Array, 'samp window']
    """The weight of each sample on each block of its window."""
    values: Float[Array, 'k samp']
    """The sub-basis functions, shared by every block."""
    n_blocks: int = field(metadata={'static': True})
    """The number of blocks."""

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n_blocks, self.values.shape[-2])

    @property
    def _array_dtype(self) -> DTypeLike:
        return self.values.dtype

    def __post_init__(self) -> None:
        super().__post_init__()
        n_points, n_window = self.taps.shape[-2:]
        if self.offset.shape[-1] != n_points or self.values.shape[-1] != n_points:
            msg = (
                f'sample axes disagree: offset {self.offset.shape}, taps {self.taps.shape}, '
                f'values {self.values.shape}'
            )
            raise ValueError(msg)
        if n_window > self.n_blocks:
            raise ValueError(f'window ({n_window}) wider than block count ({self.n_blocks})')

    @property
    def n_points(self) -> int:
        return self.offset.shape[-1]

    @property
    def blocks(self) -> Int[Array, 'samp window']:
        """The window of blocks each sample reads."""
        window = jnp.arange(self.taps.shape[-1], dtype=self.offset.dtype)
        return self.offset[..., None] + window

    def _expand(self, coeffs: Float[Array, '*batch block k']) -> Float[Array, '*batch samp']:
        # gather each sample's window of block coefficients, contract over the sub-basis
        # index against the shared values and over the window against its taps.
        gathered = coeffs[..., self.blocks, :]  # (*batch, samp, O, k)
        return _einsum('...soj,so,js->...s', gathered, self.taps, self.values)

    def _project(self, signal: Float[Array, '*batch samp']) -> Float[Array, '*batch block k']:
        batch = signal.shape[:-1]
        signals = signal.reshape(-1, self.n_points)
        # one scatter per window slot keeps each slot's block ids sorted, as `offset` is
        sums = jnp.zeros((self.n_blocks, signals.shape[0], self.shape[1]), self.dtype)
        for o in range(self.taps.shape[1]):  # window width is static
            slot = self.taps[:, o] * self.values  # (k, samp)
            contrib = signals.T[:, :, None] * slot.T[:, None, :]  # (samp, batch, k)
            sums = sums + jax.ops.segment_sum(
                contrib, self.offset + o, self.n_blocks, indices_are_sorted=True
            )
        return jnp.moveaxis(sums, 1, 0).reshape(*batch, *self.shape)

    def _gram(self, weights: Float[Array, ' samp']) -> Float[Array, 'n w1 k k']:
        # A sample adds `taps[o] · taps[o + d] · weight · v_a · v_b` to the coupling of the blocks
        # its window slots `o` and `o + d` read, which lie `d` apart: band `d`. The transient memory
        # is that of one `(samp, k)` array, never `(samp, k, k)`.
        n_window, k = self.taps.shape[1], self.values.shape[0]
        # slot `o` against slot `o + d`, for every band `d`: scanned rather than unrolled, so that
        # one pair's `(samp, k)` contribution is live at a time
        pairs = jnp.array([(o, d) for d in range(n_window) for o in range(n_window - d)])

        def row(weighted: Float[Array, ' samp']) -> Float[Array, 'n w1 k']:
            """Every band of the Gram's rows for one sub-basis function `a`, `weighted = w · v_a`."""

            def add(bands: Array, pair: Array) -> tuple[Array, None]:
                o, d = pair[0], pair[1]
                taps = self.taps[:, o] * self.taps[:, o + d] * weighted
                # The transpose stays here, where it fuses into the product: hoisted out of the
                # scan, it would copy the values.
                block = jax.ops.segment_sum(
                    (self.values * taps).T, self.offset + o, self.n_blocks, indices_are_sorted=True
                )
                return bands.at[:, d].add(block), None

            zeros = jnp.zeros((self.n_blocks, n_window, k), self.values.dtype)
            return jax.lax.scan(add, zeros, pairs)[0]

        rows = jax.lax.map(row, self.values * weights)  # (k_a, n, w1, k_b)
        return jnp.moveaxis(rows, 0, 2)

    @property
    def _n_blocks(self) -> int:
        return self.n_blocks

    def support(self) -> 'WindowedBasis':
        return self


def _single_block(values: Float[Array, 'k samp']) -> WindowedBasis:
    """A dense basis as a [`WindowedBasis`][]: one block, every sample in it with a unit tap."""
    n_points = values.shape[-1]
    taps = jnp.ones((n_points, 1), values.dtype)
    return WindowedBasis(jnp.zeros(n_points, jnp.int32), taps, values, 1)


class DecimatedBasis(Basis):
    """A basis stored on a coarser time grid, constant over each block of samples.

    Wraps any basis evaluated on the coarse grid, `inner`, of `ceil(n_samples / decimation_factor)`
    points. Synthesis holds each coarse value over its block of `decimation_factor` samples,
    trimming the last block to `n_samples`, and analysis (the transpose) sums each block back
    down; the two are exact transposes. Hold/block-sum is a zeroth-order resampler, exact only for
    content well below the coarse-grid Nyquist frequency `sample_rate / (2 * decimation_factor)`, so
    choose `decimation_factor` to keep that above the template's band edge.

    The amplitudes, and the batch of bases, are those of `inner`.

    Examples:
        >>> inner = TensorBasis(jnp.array([[1.0, 2.0, 3.0]]))
        >>> basis = DecimatedBasis(inner, decimation_factor=2, n_samples=5)
        >>> basis(jnp.array([1.0]))
        Array([1., 1., 2., 2., 3.], dtype=float32)
        >>> basis.T(jnp.arange(5.0))
        Array([23.], dtype=float32)
    """

    inner: Basis
    decimation_factor: int = field(metadata={'static': True})
    n_samples: int = field(metadata={'static': True})

    def __post_init__(self) -> None:
        if self.decimation_factor < 1:
            raise ValueError(f'decimation_factor must be >= 1, got {self.decimation_factor}')
        n_dec, factor = self.inner.n_points, self.decimation_factor
        # the coarse grid is the count of blocks covering the samples
        if not (n_dec - 1) * factor < self.n_samples <= n_dec * factor:
            msg = (
                f'inner basis of {n_dec} points does not cover {self.n_samples} samples, '
                f'decimation_factor={factor}'
            )
            raise ValueError(msg)
        # the batch is the inner basis's
        if self.batch_shape and self.batch_shape != self.inner.batch_shape:
            msg = f'batch_shape {self.batch_shape} differs from the inner {self.inner.batch_shape}'
            raise ValueError(msg)
        object.__setattr__(self, 'batch_shape', self.inner.batch_shape)
        super().__post_init__()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.inner.shape

    @property
    def _array_dtype(self) -> DTypeLike:
        return self.inner.dtype

    @property
    def n_points(self) -> int:
        return self.n_samples

    def _with_batch_shape(self, batch_shape: tuple[int, ...]) -> Self:
        inner = self.inner._with_batch_shape(batch_shape)
        return type(self)(inner, self.decimation_factor, self.n_samples)

    def _upsample(self, x: Float[Array, '*batch dec']) -> Float[Array, '*batch samp']:
        return jnp.repeat(x, self.decimation_factor, axis=-1)[..., : self.n_samples]

    def _downsample(self, s: Float[Array, '*batch samp']) -> Float[Array, '*batch dec']:
        # adjoint of `_upsample`: the tail is padded to a whole block
        n_dec, factor = self.inner.n_points, self.decimation_factor
        s = jnp.pad(s, [(0, 0)] * (s.ndim - 1) + [(0, n_dec * factor - self.n_samples)])
        return s.reshape(*s.shape[:-1], n_dec, factor).sum(axis=-1)

    def _expand(self, coeffs: Float[Array, '...']) -> Float[Array, '*batch samp']:
        return self._upsample(self.inner(coeffs))

    def _project(self, signal: Float[Array, '*batch samp']) -> Float[Array, '...']:
        return self.inner.T(self._downsample(signal))

    def _gram(self, weights: Float[Array, ' samp']) -> Float[Array, 'n w1 k k']:
        # each sample reads exactly one coarse point, so Uᵀ diag(w) U = diag(block-summed w), with
        # U the hold upsampling: the Gram never leaves the coarse grid
        return self.inner.gram(self._downsample(weights))

    @property
    def _n_blocks(self) -> int:
        return self.inner._n_blocks

    def support(self) -> WindowedBasis:
        # each sample reads the inner support at its coarse point
        support = self.inner.support()
        coarse = jnp.arange(self.n_samples) // self.decimation_factor
        return WindowedBasis(
            support.offset[coarse],
            support.taps[coarse],
            support.values[:, coarse],
            support.n_blocks,
        )
