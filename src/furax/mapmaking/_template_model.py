"""Templates of structured signals in the TOD, and the operator that sums them.

A [`Template`][] is a [`Basis`][furax.mapmaking.bases.Basis] of functions of time, seen by each
detector and stream of a TOD through a coupling, with independent amplitudes along some of
these axes. Amplitudes per detector or per stream, shared across them, or seen through a pattern
over the focal plane are all choices of these ingredients. As an operator, a template maps its
amplitudes to the TOD that a [`TODFormat`][furax.obs.TODFormat] describes, and
[`TemplateOperator`][] sums the templates of one TOD.
"""

from dataclasses import field
from functools import reduce
from math import prod
from typing import Literal, NamedTuple, get_args

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, PyTree

from furax import AbstractLinearOperator
from furax.core import BlockRowOperator, TransposeOperator
from furax.obs import TOD, TODFormat

from .bases import Basis, _einsum

__all__ = [
    'Template',
    'TemplateOperator',
]

PerAxis = Literal['stream', 'det']
"""An axis along which a template has independent sets of amplitudes."""

_PER_AXES: tuple[PerAxis, ...] = get_args(PerAxis)
"""The axes of `per`, in the order the amplitudes carry them."""


class Template(AbstractLinearOperator):
    r"""Functions of time, seen by each detector and stream of a TOD through a coupling.

    Stream $l$ of detector $d$ sees

    $$ s_{ld}(t) = \sum_m C_{ldm} \sum_k b_{dk}(t)\, a_{o(l, d)\, km}, $$

    with $b_{dk}$ the basis functions, $C_{ldm}$ the coupling to mode $m$, and $a_{o(l, d)}$ the
    amplitude set it sees. `per` names the axes with independent sets:

    - `'stream'`: one set per stream of `streams`;
    - `'det'`: one set per detector, or per group of detectors (such as a wafer) with
      `det_groups`.

    Along the other axes, the amplitudes are shared: `per=()` fits one set for the whole TOD.
    The amplitudes have shape `(*sets, *basis.shape, *modes)`: the sets along `per` (streams, then
    the detector axes or groups), the basis shape, and the modes if there is a coupling. The
    template maps them to the whole TOD, zero on the streams it does not cover.

    Raises:
        ValueError: If `per`, `streams`, the coupling or `det_groups` are malformed or do not fit
            the TOD (a coupling axis may also have size 1), or if the basis differs from the TOD
            in detector batch, sample count or dtype.

    Examples:
        A common mode, one offset per block of 4 samples, seen by 3 detectors with known gains,
        the coupling. With `per=()`, the detectors share one timestream of offsets:

        >>> from furax.mapmaking.bases import OffsetBasis, TensorBasis
        >>> tod = TODFormat((3, 16), jnp.float32)
        >>> offsets = OffsetBasis(4, 16, jnp.float32)
        >>> gains = jnp.array([[1.0], [0.5], [-1.0]])  # (det, mode)
        >>> Template(offsets, tod, per=(), coupling=gains).in_structure.shape  # (block, mode)
        (4, 1)

        With `per=('det',)`, each detector fits its own offsets, which the gains only rescale:

        >>> Template(offsets, tod, per=('det',), coupling=gains).in_structure.shape
        (3, 4, 1)

        With `det_groups`, the detectors of each wafer share their offsets, here the first two
        detectors on one wafer and the third on another:

        >>> wafers = jnp.array([[True, False], [True, False], [False, True]])  # (det, group)
        >>> Template(offsets, tod, per=('det',), det_groups=wafers).in_structure.shape
        (2, 4)

        A temperature-to-polarization leakage, one coefficient per stream and detector on Q and U.
        The basis is each detector's own temperature stream:

        >>> qu = TODFormat((3, 16), jnp.float32, stokes='QU')
        >>> temperature = TensorBasis(jnp.ones((3, 1, 16)), batch_shape=(3,))
        >>> Template(temperature, qu, streams='qu').in_structure.shape  # (stream, det, 1)
        (2, 3, 1)

        The same leakage as a polarization $a$ uniform over the focal plane, common to Q and U: a
        detector of polarization angle $\gamma$ sees $\mathrm{Re}(a\, e^{2i\gamma})$ on Q and
        $\mathrm{Im}(a\, e^{2i\gamma})$ on U, a coupling to the modes $(\mathrm{Re}\, a,
        \mathrm{Im}\, a)$:

        >>> gamma = jnp.array([0.0, 0.4, 1.2])
        >>> cos, sin = jnp.cos(2 * gamma), jnp.sin(2 * gamma)
        >>> spin2 = jnp.stack([jnp.stack([cos, -sin], -1), jnp.stack([sin, cos], -1)])
        >>> spin2.shape  # (stream, det, mode)
        (2, 3, 2)
        >>> Template(temperature, qu, streams='qu', per=(), coupling=spin2).in_structure.shape
        (1, 2)
    """

    basis: Basis
    """The functions of time: one basis, or one per detector (`batch_shape=tod.det_shape`)."""
    tod: TODFormat = field(metadata={'static': True})
    """The TOD the template adds to."""
    streams: str | None = field(default=None, metadata={'static': True})
    """The streams covered, as Stokes parameters in lower case, such as `'qu'`; `None` unless the
    TOD is demodulated."""
    per: tuple[PerAxis, ...] = field(default=('stream', 'det'), metadata={'static': True})
    """The axes with independent amplitude sets, among `'stream'` and `'det'`."""
    coupling: Float[Array, '*stream *det mode'] | None = None
    """How each stream of each detector sees each mode: `(n_streams, *det_shape, n_modes)`, without
    the stream axis unless the TOD is demodulated. An axis of size 1 applies to all streams or
    detectors along it. `None` is a single mode of unit coupling, with no mode axis."""
    det_groups: Bool[Array, '*det group'] | None = None
    """The one-hot group of each detector, `(*det_shape, n_groups)`, for detectors that share
    their amplitudes along `'det'`; `None` gives each detector its own."""

    def __post_init__(self) -> None:
        # `in_structure` is derived, never passed: it can then never disagree with the arrays.
        if self.in_structure is not None:
            raise ValueError('in_structure is derived from the template; do not pass it')
        if len(set(self.per)) != len(self.per) or not set(self.per) <= set(_PER_AXES):
            raise ValueError(f'per takes distinct axes among {_PER_AXES}, got {self.per}')
        # The amplitudes carry the axes of `per` in a fixed order, whatever order they are given in;
        # only a demodulated TOD has a stream axis, and a single timestream has no detector axis.
        per = tuple(
            axis
            for axis in _PER_AXES
            if axis in self.per
            and (axis != 'stream' or self.streams is not None)
            and (axis != 'det' or self.tod.det_shape)
        )
        object.__setattr__(self, 'per', per)
        self._check_streams()
        self._check_arrays()
        structure = jax.ShapeDtypeStruct(self._amplitude_shape(), self.tod.dtype)
        object.__setattr__(self, 'in_structure', structure)
        super().__post_init__()

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        # Read off static metadata rather than traced through `mv`, which would see the extra
        # leading axis of a template stacked across observations.
        return self.tod.structure

    @property
    def n_det_groups(self) -> int | None:
        """The number of detector groups; `None` without `det_groups`."""
        return None if self.det_groups is None else self.det_groups.shape[-1]

    @property
    def per_detector(self) -> bool:
        """Whether every detector has its own set of amplitudes, and no other detector sees it."""
        return 'det' in self.per and self.det_groups is None

    def mv(self, x: Array) -> TOD:
        return _sum_into(self.tod, [(self._stream_indices, self._expand_streams(x))])

    def transpose(self) -> AbstractLinearOperator:
        return _TemplateTranspose(self)

    def _check_streams(self) -> None:
        streams, stokes = self.streams, self.tod.stokes
        if streams is None:
            if stokes is not None:
                raise ValueError(f'the template covers no stream of stokes={stokes!r}')
            return
        if not streams or streams != streams.lower() or len(set(streams)) != len(streams):
            raise ValueError(f'streams must be distinct and lower case, got {streams!r}')
        if extra := sorted(set(streams) - set((stokes or '').lower())):
            raise ValueError(f'the template covers streams {extra} outside stokes={stokes!r}')

    @property
    def _stream_indices(self) -> tuple[int, ...]:
        """The indices along the TOD's stream axis of the streams covered."""
        if self.streams is None:
            return (0,)
        assert self.tod.stokes is not None  # streams are those of a demodulated TOD
        return tuple(self.tod.stokes.lower().index(stream) for stream in self.streams)

    def _check_arrays(self) -> None:
        tod = self.tod
        det_shape = tod.det_shape
        if self.det_groups is not None and (
            'det' not in self.per or self.det_groups.shape[:-1] != det_shape
        ):
            msg = f"det_groups, of shape (*{det_shape}, group), groups the amplitudes along 'det'"
            raise ValueError(msg)
        if self.coupling is not None:
            shape = self.coupling.shape
            stream_axis = () if self.streams is None else (f'1 or {len(self.streams)}',)
            expected = f'({", ".join([*stream_axis, *map(str, det_shape)])}, mode)'
            ndim = len(stream_axis) + len(det_shape) + 1
            n_dets = shape[len(stream_axis) : -1]
            if (
                len(shape) != ndim
                or (self.streams is not None and shape[0] not in {1, len(self.streams)})
                or any(n not in {1, full} for n, full in zip(n_dets, det_shape, strict=True))
            ):
                msg = f'the coupling has shape {expected}, an axis possibly of size 1, got {shape}'
                raise ValueError(msg)
        basis = self.basis
        if basis.batch_shape not in {(), det_shape}:
            msg = f'the basis has a batch {basis.batch_shape}, not the detectors {det_shape}'
            raise ValueError(msg)
        if basis.n_points != tod.n_samples:
            raise ValueError(f'the basis has {basis.n_points} samples, not {tod.n_samples}')
        if jnp.dtype(basis.dtype) != tod.dtype:
            raise ValueError(f'the basis has dtype {basis.dtype}, not {tod.dtype}')

    def _amplitude_shape(self) -> tuple[int, ...]:
        sets = {'stream': (len(self._stream_indices),), 'det': self.tod.det_shape}
        if self.n_det_groups is not None:
            sets['det'] = (self.n_det_groups,)
        modes = () if self.coupling is None else self.coupling.shape[-1:]
        return (*(n for axis in self.per for n in sets[axis]), *self.basis.shape, *modes)

    # Internally, the amplitudes take a canonical layout with fixed axes, whatever `per` and the
    # coupling: `(stream set, detector set, *basis.shape, mode)`, an axis of size 1 where the sets
    # are shared, and a single mode without coupling. The detector axes are flattened into one, here
    # as in the signals, `(detector, samp)`. The amplitudes stream `l` sees are then the rows `[l]`:
    # one row per detector, per group, or a single one. Detector `d` sees row `o` through the weight
    # `W[d, o, m]`, the coupling of mode `m` where `d` belongs to `o`, and zero elsewhere. With a
    # set per detector, the weight is diagonal: the coupling alone.

    @property
    def _canonical_shape(self) -> tuple[int, ...]:
        """The shape of the amplitudes in the canonical layout."""
        n_sets = 1
        if 'det' in self.per:
            n_sets = self.tod.n_dets if self.n_det_groups is None else self.n_det_groups
        n_modes = 1 if self.coupling is None else self.coupling.shape[-1]
        n_streams = len(self._stream_indices) if 'stream' in self.per else 1
        return (n_streams, n_sets, *self.basis.shape, n_modes)

    def _to_canonical(self, a: Array) -> Array:
        """The amplitudes in the canonical layout: only axes of size 1 differ, or are merged."""
        return a.reshape(self._canonical_shape)

    def _from_canonical(self, a: Array) -> Array:
        """The inverse of `_to_canonical`."""
        return a.reshape(self.in_structure.shape)

    def _expand_streams(self, a: Array) -> list['Array | _DenseTerm']:
        """The signal on each stream covered, `(n_dets, samp)`, or `(1, samp)` if the same on all.

        A template with a set per detector on a dense shared basis gives its coefficients and
        basis matrix instead, for the sum to apply in one product with others alike. Streams that
        all see the same signal share one array.
        """
        n_streams = len(self._stream_indices)
        if self.per_detector and (dense := self.basis._dense_values()) is not None:
            terms = []
            for stream in range(n_streams):
                coefficients = self._coefficients(self._rows(a, stream), stream)
                terms.append(_DenseTerm(coefficients.reshape(self.tod.n_dets, -1), dense))
            return terms
        if self._stream_free:
            return n_streams * [self._expand(a, 0)]
        return [self._expand(a, stream) for stream in range(n_streams)]

    def _project_streams(self, streams: list[Float[Array, 'det samp']]) -> Array:
        """The transpose of `_expand_streams`: the amplitudes from each stream's signal."""
        if self._stream_free:  # project the sum of the streams once
            rows = self._project(reduce(jnp.add, streams), 0)[None]
        else:
            parts = [self._project(signal, stream) for stream, signal in enumerate(streams)]
            rows = jnp.stack(parts) if 'stream' in self.per else reduce(jnp.add, parts)[None]
        return self._from_canonical(rows)

    @property
    def _stream_free(self) -> bool:
        """Whether every stream sees the same amplitudes through the same coupling."""
        return 'stream' not in self.per and (
            self.coupling is None or self.streams is None or self.coupling.shape[0] == 1
        )

    @property
    def _in_set_space(self) -> bool:
        """Whether to expand the basis once per (set, mode), then weight it per detector.

        So it is when the detectors share the basis but not one set each: the expansion then does
        not scale with the number of detectors.
        """
        return not self.basis.batch_shape and not self.per_detector

    def _rows(self, a: Array, stream: int) -> Array:
        """The amplitudes stream number `stream` sees, as rows `(set, *basis.shape, mode)`."""
        return self._to_canonical(a)[self._stream_set(stream)]

    def _stream_set(self, stream: int) -> int:
        """The stream set, in the canonical layout, that stream number `stream` sees."""
        return stream if 'stream' in self.per else 0

    def _expand(self, a: Array, stream: int) -> Array:
        """The signal on stream number `stream`: `(n_dets or 1, samp)`, 1 if the same on all."""
        rows = self._rows(a, stream)
        if self._in_set_space:
            expanded = self.basis(jnp.moveaxis(rows, -1, 1))  # (set, mode, samp)
            return _einsum('dom,oms->ds', self._weights(stream), expanded)
        return self._expand_basis(self._coefficients(rows, stream))

    def _project(self, signal: Float[Array, 'det samp'], stream: int) -> Array:
        """The transpose of `_expand` on stream `stream`: rows `(set, *basis.shape, mode)`."""
        if self._in_set_space:
            weighted = _einsum('ds,dom->oms', signal, self._weights(stream))
            return jnp.moveaxis(self.basis.T(weighted), 1, -1)
        if self.per_detector:
            return _einsum('d...,dm->d...m', self._project_basis(signal), self._coupling_on(stream))
        return _einsum('d...,dom->o...m', self._project_basis(signal), self._weights(stream))

    def _coefficients(self, rows: Array, stream: int) -> Array:
        """Each detector's coefficients on the basis: `(n_dets or 1, *basis.shape)`."""
        if self.per_detector:
            return _einsum('d...m,dm->d...', rows, self._coupling_on(stream))
        return _einsum('o...m,dom->d...', rows, self._weights(stream))

    def _expand_basis(self, coefficients: Array) -> Array:
        """The basis on coefficients `(n_dets or 1, *basis.shape)`: `(n_dets or 1, samp)`."""
        if not self.basis.batch_shape:
            return self.basis(coefficients)
        tod = self.tod
        shaped = jnp.broadcast_to(coefficients, (tod.n_dets, *coefficients.shape[1:]))
        signal = self.basis(shaped.reshape(*tod.det_shape, *self.basis.shape))
        return signal.reshape(tod.n_dets, tod.n_samples)

    def _project_basis(self, signal: Float[Array, 'det samp']) -> Array:
        """The transpose of `_expand_basis` for `n_dets` rows."""
        if not self.basis.batch_shape:
            return self.basis.T(signal)
        tod = self.tod
        coefficients = self.basis.T(signal.reshape(tod.shape))
        return coefficients.reshape(tod.n_dets, *self.basis.shape)

    def _weights(self, stream: int) -> Float[Array, 'det set mode']:
        """`W[d, o, m]` on stream number `stream`, for templates without a set per detector."""
        coupling = self._coupling_on(stream)[:, None, :]  # every detector sees the single set
        if self.det_groups is None:
            return coupling
        groups = self.det_groups.reshape(self.tod.n_dets, -1)
        return groups[..., None] * coupling

    def _coupling_on(self, stream: int) -> Float[Array, 'det mode']:
        """The coupling on stream number `stream`: `(n_dets or 1, n_modes)`."""
        if self.coupling is None:
            return jnp.ones((1, 1), self.tod.dtype)
        coupling = self.coupling
        if self.streams is not None:
            coupling = coupling[stream if coupling.shape[0] > 1 else 0]
        if prod(coupling.shape[:-1]) == 1:  # the same on every detector
            return coupling.reshape(1, -1)
        shape = (*self.tod.det_shape, coupling.shape[-1])
        return jnp.broadcast_to(coupling, shape).reshape(self.tod.n_dets, -1)


class _TemplateTranspose(TransposeOperator):
    operator: Template

    def mv(self, x: TOD) -> Array:
        template = self.operator
        tod = template.tod
        streams = tod.as_array(x).reshape(tod.n_streams, tod.n_dets, tod.n_samples)
        return template._project_streams([streams[i] for i in template._stream_indices])


class TemplateOperator(BlockRowOperator):
    r"""The sum of templates of one TOD, from amplitudes keyed by template name.

    The block row of the templates, $T = [T_1 \mid T_2 \mid \dots]$. Its transpose is the block
    column of theirs.

    Raises:
        ValueError: If there is no template, or if the templates differ in TOD.

    Examples:
        A polynomial per stream and detector, and a leakage on Q and U of one shared amplitude, seen
        with a gain per stream and detector:

        >>> from furax.mapmaking.bases import TensorBasis
        >>> tod = TODFormat((3, 5), jnp.float32, stokes='IQU')
        >>> poly = Template(TensorBasis(jnp.ones((2, 5))), tod, streams='iqu')
        >>> gains = jnp.ones((2, 3, 1))  # (stream, det, mode)
        >>> constant = TensorBasis(jnp.ones((1, 5)))
        >>> leak = Template(constant, tod, streams='qu', per=(), coupling=gains)
        >>> op = TemplateOperator({'poly': poly, 'leak': leak})
        >>> {name: s.shape for name, s in op.in_structure.items()}
        {'leak': (1, 1), 'poly': (3, 3, 2)}
    """

    blocks: dict[str, Template]

    def __init__(self, templates: dict[str, Template]) -> None:
        if not templates:
            raise ValueError('a template operator needs at least one template')
        if len({template.tod for template in templates.values()}) > 1:
            raise ValueError('the templates add to different TODs')
        super().__init__(templates)

    @property
    def templates(self) -> dict[str, Template]:
        """The templates, by name."""
        return self.blocks

    @property
    def tod(self) -> TODFormat:
        """The TOD the templates add to."""
        return next(iter(self.templates.values())).tod

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.tod.structure

    def mv(self, x: dict[str, Array]) -> TOD:
        # The block row would add up each template's whole TOD. Instead, each stream of the TOD sums
        # the signals of the templates covering it, the dense ones expanded in one product.
        terms = [
            (t._stream_indices, t._expand_streams(x[name])) for name, t in self.templates.items()
        ]
        return _sum_into(self.tod, terms)


class _DenseTerm(NamedTuple):
    """A signal as each detector's coefficients on a dense basis shared by the detectors."""

    coefficients: Float[Array, 'det k']
    values: Float[Array, 'k samp']


def _sum_into(tod: TODFormat, terms: list[tuple[tuple[int, ...], list[Array | _DenseTerm]]]) -> TOD:
    """The TOD summing the signals of templates, given with the streams they cover, by index."""
    by_stream: list[list[Array | _DenseTerm]] = [[] for _ in range(tod.n_streams)]
    for streams, signals in terms:
        for stream, signal in zip(streams, signals, strict=True):
            by_stream[stream].append(signal)
    streams = jnp.stack([_sum_stream(tod, signals) for signals in by_stream])
    return tod.from_array(streams.reshape(tod.n_streams, *tod.shape))


def _sum_stream(tod: TODFormat, signals: list[Array | _DenseTerm]) -> Float[Array, 'det samp']:
    """One stream's sum of signals, `(n_dets, samp)`."""
    stream = jnp.zeros((tod.n_dets, tod.n_samples), tod.dtype)
    dense = [signal for signal in signals if isinstance(signal, _DenseTerm)]
    for signal in signals:
        if not isinstance(signal, _DenseTerm):
            stream = stream + signal
    if dense:
        # Each template would otherwise make its own pass over the stream: the dense ones are
        # stacked into one `(K, samp)` matrix, applied in one product over all detectors.
        coefficients = jnp.concatenate([term.coefficients for term in dense], axis=1)
        values = jnp.concatenate([term.values for term in dense])
        stream = stream + _einsum('dk,ks->ds', coefficients, values)
    return stream
