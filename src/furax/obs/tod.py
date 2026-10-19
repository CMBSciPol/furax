"""Time-ordered data (TOD) format conventions."""

from dataclasses import dataclass
from math import prod

import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Key, PyTree, ScalarLike, Shaped

from furax.core import DiagonalOperator
from furax.obs.stokes import Stokes, ValidStokesLiteral
from furax.tree import full_like, normal_like, uniform_like

__all__ = [
    'TOD',
    'TODFormat',
]

type TOD = Array | Stokes
"""Possible types for the time-ordered data container."""


@dataclass(frozen=True)
class TODFormat:
    """Format of time-ordered data (TOD).

    The TOD is usually a plain array of `shape`, with the samples last. The axes before the samples
    index the detectors: one axis usually, none for a single timestream, several for more general
    sample data.

    In the case of demodulated data, Stokes parameters are separated: the TOD is then stored in a
    [`Stokes`][furax.obs.stokes.Stokes] container, i.e. one stream per Stokes parameter.

    Examples:
        >>> tod_format = TODFormat((3, 100), jnp.float32, stokes='QU')
        >>> tod_format.det_shape, tod_format.n_dets, tod_format.n_samples
        ((3,), 3, 100)
        >>> tod_format.zeros().q.shape
        (3, 100)
        >>> tod_format.n_streams, tod_format.as_array(tod_format.zeros()).shape
        (2, (2, 3, 100))
        >>> weights = jnp.stack([jnp.ones((3, 100)), 2 * jnp.ones((3, 100))]).astype(jnp.float32)
        >>> tod_format.diagonal(weights)(tod_format.ones()).u[0, 0]
        Array(2., dtype=float32)
    """

    shape: tuple[int, ...]
    """The shape of the TOD, or of one of its streams: `(*det, samp)`."""
    dtype: DTypeLike
    """The dtype of the TOD."""
    stokes: ValidStokesLiteral | None = None
    """The Stokes parameters of a demodulated TOD, such as `'IQU'`; `None` otherwise."""

    def __post_init__(self) -> None:
        if not self.shape:
            raise ValueError('the TOD shape has at least the sample axis')
        if self.stokes is not None:
            Stokes.class_for(self.stokes)  # raises for an invalid combination
        object.__setattr__(self, 'shape', tuple(int(n) for n in self.shape))
        object.__setattr__(self, 'dtype', jnp.dtype(self.dtype))

    @property
    def det_shape(self) -> tuple[int, ...]:
        """The detector axes."""
        return self.shape[:-1]

    @property
    def n_dets(self) -> int:
        """The number of detectors, over every detector axis."""
        return prod(self.det_shape)

    @property
    def n_samples(self) -> int:
        return self.shape[-1]

    @property
    def n_streams(self) -> int:
        """The number of streams: one per Stokes parameter if demodulated, else a single one."""
        return 1 if self.stokes is None else len(self.stokes)

    @property
    def structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        """The structure of the TOD."""
        if self.stokes is None:
            return jax.ShapeDtypeStruct(self.shape, self.dtype)
        return Stokes.class_for(self.stokes).structure_for(self.shape, self.dtype)

    def as_array(self, tod: TOD) -> Shaped[Array, 'stream *det samp']:
        """The TOD as one array, its streams stacked on a leading axis: `(n_streams, *shape)`."""
        return tod.data if isinstance(tod, Stokes) else tod[None]

    def from_array(self, array: Shaped[Array, 'stream *det samp']) -> TOD:
        """The inverse of `as_array`: the TOD whose streams `array` stacks."""
        if self.stokes is None:
            return self._leaves(array)
        return Stokes.class_for(self.stokes).from_array(array)

    def diagonal(self, weights: Shaped[Array, 'stream *det samp']) -> DiagonalOperator:
        """The diagonal operator on the TOD that scales sample `(s, d, t)` by `weights[s, d, t]`."""
        return DiagonalOperator(self._leaves(weights), in_structure=self.structure)

    def full(self, fill_value: ScalarLike) -> TOD:
        """A TOD filled with `fill_value`."""
        return full_like(self.structure, fill_value)

    def zeros(self) -> TOD:
        """A TOD of zeros."""
        return self.full(0)

    def ones(self) -> TOD:
        """A TOD of ones."""
        return self.full(1)

    def normal(self, key: Key[Array, '']) -> TOD:
        """A TOD of independent standard normal samples."""
        return normal_like(self.structure, key)

    def uniform(self, key: Key[Array, ''], minval: float = 0.0, maxval: float = 1.0) -> TOD:
        """A TOD of independent samples, uniform in `[minval, maxval)`."""
        return uniform_like(self.structure, key, minval, maxval)

    def _leaves(self, array: Shaped[Array, 'stream *det samp']) -> Shaped[Array, '...']:
        """`array` as laid out over the leaves of the structure: with no stream axis unless demodulated."""
        return array if self.stokes is not None else array[0]
