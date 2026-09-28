import copy
import dataclasses
import math
from dataclasses import field
from typing import NamedTuple, Self, TypeVar

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jax import jit, lax
from jaxtyping import Array, Bool, Float, Int, PyTree

from furax import AbstractLinearOperator
from furax.core import TransposeOperator
from furax.core.rules import AbstractCompositionRule, NoReduction
from furax.obs.landscapes import StokesLandscape
from furax.obs.operators._qu_rotations import (
    QURotationOperator,
    QURotationTransposeOperator,
    Spin2Rotation,
)
from furax.obs.sampling import (
    AbstractSampler,
    DiscretizedBeam,
    PointingRows,
    PolarizationFrame,
    PrecomputedSampler,
    QuaternionSampler,
    SampleIndex,
    SamplingKernel,
)
from furax.obs.spin2 import rotated_gather, rotated_scatter
from furax.obs.stencil import Interpolation
from furax.obs.stokes import Stokes

__all__ = [
    'PointingOperator',
]

_StokesT = TypeVar('_StokesT', bound=Stokes)

# Default number of samples per batch, per backend
#
# Intermediates grow with the number of samples per batch. These sizes were about the fastest in
# benchmarks of CAR and HEALPix maps, nearest and bilinear (CPU: Intel Core i5-14400F, GPU: NVIDIA
# GeForce RTX 4060).
# On GPU, larger batches are slower because the intermediates no longer fit in the device cache.
# On CPU, larger batches seem to keep getting faster, up to 1.5x for a single (nearest) batch.
# 2^21 caps the intermediates at a few hundred MB.
_GPU_BATCH_SAMPLES = 2**17
_CPU_BATCH_SAMPLES = 2**21


class PointingOperator(AbstractLinearOperator):
    r"""Operator that samples a sky map, e.g. into time-ordered data (TOD).

    Where every sample reads the map is given by a
    [`AbstractSampler`][furax.obs.sampling.AbstractSampler], as one sparse row of the pointing
    matrix per sample: the pixels it reads, their weights, and the rotation of each pixel's
    $(Q, U)$ into the frame the sample is returned in, e.g. that of a detector. The operator loops
    over the samples in batches of `batch_samples` consecutive samples, in the row-major order of
    the samples' shape, and, for each batch, gathers the pixels, rotates and weighs them, then
    multiplies the samples by the sampler's scaling, if it has one. A batch may span several
    detectors, or part of one.

    The transpose accumulates samples into a sky map (binning), and is exact whatever the sampler.

    Build one from quaternions with [`PointingOperator.create`][], or from any sampler with
    [`PointingOperator.from_sampler`][]. [`PointingOperator.precomputed`][] computes the pointing
    once, for a map read many times.

    Attributes:
        landscape: The sky pixelization.
        sampler: Where each sample reads the map.
        batch_samples: Number of samples processed per batch. Leave `None` for a backend-dependent
            default; set to 0 to process them all at once.
    """

    landscape: StokesLandscape
    sampler: AbstractSampler
    batch_samples: int | None = field(default=None, metadata={'static': True})

    @classmethod
    def create(
        cls,
        landscape: StokesLandscape,
        boresight_quaternions: Quaternion,
        detector_quaternions: Quaternion,
        *,
        batch_samples: int | None = None,
        frame: PolarizationFrame = 'boresight',
        interpolate: bool = False,
        beam: DiscretizedBeam | None = None,
    ) -> 'PointingOperator':
        r"""Build the operator from the boresight pointing and the detector offsets.

        Args:
            landscape: The sky pixelization.
            boresight_quaternions: Boresight quaternions, shape (n_samples,).
            detector_quaternions: Detector offset quaternions, shape (n_detectors,).
            batch_samples: Number of samples processed per batch, see [`PointingOperator`][].
            frame: The basis Q and U are returned in, see
                [`QuaternionSampler`][furax.obs.sampling.QuaternionSampler].
            interpolate: If True, bilinear interpolation over the four nearest pixels, otherwise
                nearest neighbour.
            beam: The beam each sample integrates over, centred on the detector's line of sight,
                or `None` to read the line of sight alone. Per-component weights act on the
                components of the output, i.e. in the polarization frame chosen by `frame`, not in
                the sky's meridian basis.

        Examples:
            A single detector at the boresight, pointing at random directions given as the ZYZ
            Euler angles $(\phi, \theta, \psi)$ of the boresight:

            >>> import jax
            >>> import jax.numpy as jnp
            >>> from fastquat import Quaternion
            >>> from furax.math.coords import IsoAngles
            >>> from furax.obs.landscapes import HealpixLandscape
            >>> theta, phi, psi = jax.random.uniform(jax.random.key(0), (3, 1000)) * jnp.array(
            ...     [[jnp.pi], [2 * jnp.pi], [2 * jnp.pi]]
            ... )
            >>> landscape = HealpixLandscape(nside=64, stokes='IQU')
            >>> pointing = PointingOperator.create(
            ...     landscape, IsoAngles(theta, phi, psi).to_quaternion(), Quaternion.ones((1,))
            ... )
            >>> pointing.out_structure.shape
            (1, 1000)
        """
        kernel = SamplingKernel(
            interpolation=Interpolation.BILINEAR if interpolate else Interpolation.NEAREST,
            beam=None if beam is None else beam.checked(landscape, detector_quaternions.shape[0]),
        )
        sampler = QuaternionSampler(
            kernel=kernel,
            qbore=boresight_quaternions,
            qdet=detector_quaternions,
            frame=frame,
        )
        return cls.from_sampler(landscape, sampler, batch_samples=batch_samples)

    @classmethod
    def from_sampler(
        cls,
        landscape: StokesLandscape,
        sampler: AbstractSampler,
        *,
        batch_samples: int | None = None,
    ) -> 'PointingOperator':
        """Build the operator reading a map where a sampler says.

        Args:
            landscape: The sky pixelization.
            sampler: Where each sample reads the map.
            batch_samples: Number of samples processed per batch, see [`PointingOperator`][].
        """
        return cls(landscape, sampler, batch_samples, in_structure=landscape.structure)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.landscape.structure_for(self.sampler.shape)

    @jit
    def mv(self, x: _StokesT) -> _StokesT:
        """Performs the 'un-pointing' operation, i.e. map->tod."""
        x_flat = x.ravel()
        shape = self.sampler.shape
        tiling = _Tiling.plan(shape, self.batch_samples)
        if tiling.n_batches == 1:
            # a single batch needs no loop, nor a copy into the output
            return self._sample(x_flat, self.sampler.every_sample())

        # NB: lax.map over the batches is slower than this loop, on GPU and CPU alike, and holds
        # a second copy of the timestream to put its output in order
        def body(i: Int[Array, ''], tod: Array) -> Array:
            start, index, _ = tiling.batch(i)
            tod_batch = self._sample(x_flat, index)
            return lax.dynamic_update_slice(tod, tod_batch.data, (0, *start))

        # Start from an empty timestream: every slot gets overwritten by body.
        n_stokes = x.data.shape[0]
        tod = jnp.empty((n_stokes, tiling.n_rows, tiling.n_cols), x.dtype)
        tod = lax.fori_loop(0, tiling.n_batches, body, tod)
        return type(x).from_array(tod.reshape(n_stokes, *shape))

    def as_stokes_i(self, *, interpolate: bool | None = None) -> 'PointingOperator':
        """Return a copy of this operator restricted to StokesI.

        The beam is kept. Beam weights given per Stokes component reduce to those of I, or to
        their mean over the components when the operator has no I component.

        Args:
            interpolate: Override the interpolation: bilinear if True, nearest neighbour if
                False. If `None` (default), the kernel's interpolation is kept.
        """
        interpolation = self.sampler.kernel.interpolation
        if interpolate is not None:
            interpolation = Interpolation.BILINEAR if interpolate else Interpolation.NEAREST
        if self.landscape.stokes == 'I' and interpolation is self.sampler.kernel.interpolation:
            return self
        landscape = copy.copy(self.landscape)
        landscape.stokes = 'I'
        kernel = self.sampler.kernel.intensity_only()
        sampler = self.sampler.with_kernel(dataclasses.replace(kernel, interpolation=interpolation))
        return PointingOperator.from_sampler(landscape, sampler, batch_samples=self.batch_samples)

    def precomputed(self, *, batch_samples: int = 0) -> 'PointingOperator':
        """Return the same operator, with its pointing computed once.

        Hoists the quaternion-to-sky computations out of repeated applies, e.g. every iteration of
        an iterative solver, at the cost of storing the pointing:

        - With bilinear interpolation, a [`QuaternionSampler`][furax.obs.sampling.QuaternionSampler]
          stores the sky angles of every sample and beam node, see
          [`AngleSampler`][furax.obs.sampling.AngleSampler]: every apply recomputes the rows, but
          the four pixels each node reads are not stored.
        - Otherwise, the rows of the pointing matrix are stored, i.e. the pixels, weights and
          polarization rotations, see [`PrecomputedSampler`][furax.obs.sampling.PrecomputedSampler]:
          the fastest apply.

        Args:
            batch_samples: Number of samples processed per batch. The default, 0, processes them
                all at once, which is fastest once the pointing is stored.
        """
        sampler: AbstractSampler
        bilinear = self.sampler.kernel.interpolation is Interpolation.BILINEAR
        if bilinear and isinstance(self.sampler, QuaternionSampler):
            sampler = self.sampler.to_angles(self.landscape)
        else:
            sampler = PrecomputedSampler.from_sampler(self.sampler, self.landscape)
        return PointingOperator.from_sampler(self.landscape, sampler, batch_samples=batch_samples)

    def _sample(self, x_flat: _StokesT, index: SampleIndex) -> _StokesT:
        """Sample the flat map for a batch of samples, in the map's dtype."""
        tod: _StokesT
        if self._reads_nearest:
            # the gather wraps a -1 onto the last pixel, which the sample never observed
            pix = self.sampler.nearest_indices(self.landscape, index)
            tod = type(x_flat).from_array(jnp.where(pix >= 0, x_flat[pix].data, 0))
        else:
            pointing = self._pointing(index)
            tod = rotated_gather(x_flat, *pointing[:2])
            if pointing.polarization_rotation is not None:
                tod = tod.rotate_qu(*pointing.polarization_rotation)
        tod = _scaled(tod, self.sampler.scaling(index))
        # float64 pointing (rotations, scaling) promotes the samples of a float32 map
        return tod.astype(x_flat.dtype)

    def _bin(self, out: _StokesT, tod_batch: _StokesT, index: SampleIndex) -> _StokesT:
        """Scatter-add a batch of samples into the sky map `out`."""
        tod_batch = _scaled(tod_batch, self.sampler.scaling(index))
        sky_shape = self.landscape.shape
        # scatter-add per pixel while keeping the leading Stokes axis of the backing array.
        n_stokes = tod_batch.data.shape[0]
        flat = type(out).from_array(out.data.reshape(n_stokes, -1))

        if self._reads_nearest:
            pix = self.sampler.nearest_indices(self.landscape, index)
            # the scatter wraps a -1 onto the last pixel, so such a sample must add nothing
            contrib = jnp.where(pix >= 0, tod_batch.data, 0)
            binned = flat.data.at[:, pix.ravel()].add(contrib.reshape(n_stokes, -1))
        else:
            pointing = self._pointing(index)
            if pointing.polarization_rotation is not None:
                tod_batch = tod_batch.rotate_qu(*pointing.polarization_rotation.inverse())
            binned = rotated_scatter(flat, tod_batch, *pointing[:2]).data
        return type(out).from_array(binned.reshape(n_stokes, *sky_shape))

    @property
    def _reads_nearest(self) -> bool:
        """Whether every sample reads one pixel of a map without polarization.

        Such a sample needs the pixel index alone, not a stencil and rotations.
        """
        return self.sampler.kernel.reads_one_pixel and not self.landscape.has_spin2

    def _pointing(self, index: SampleIndex) -> PointingRows:
        pointing = self.sampler.pointing_rows(self.landscape, index)
        if self.landscape.has_spin2 and pointing.neighbour_rotation is None:
            raise ValueError(
                f'{type(self.sampler).__name__} returns no rotation, so it cannot read a '
                f'polarized map'
            )
        return pointing

    def rotated(self, angles: Float[Array, '...']) -> 'PointingOperator':
        """The operator followed by a rotation of Q and U by `angles`, in a single pass.

        Args:
            angles: Rotation angles in radians, broadcastable to the shape of the samples, with
                the convention of [`QURotationOperator`][furax.obs.operators.QURotationOperator].
        """
        sampler = self.sampler.rotated(Spin2Rotation.from_angles(angles))
        return PointingOperator.from_sampler(
            self.landscape, sampler, batch_samples=self.batch_samples
        )

    def transpose(self) -> AbstractLinearOperator:
        return PointingTransposeOperator(operator=self)


class PointingTransposeOperator(TransposeOperator):
    operator: PointingOperator

    @jit
    def mv(self, x: _StokesT) -> _StokesT:
        """Performs the 'pointing' operation, i.e. tod->map."""
        sampler = self.operator.sampler
        tiling = _Tiling.plan(sampler.shape, self.operator.batch_samples)
        sky_out: _StokesT = self.operator.landscape.zeros()
        if tiling.n_batches == 1:
            return self.operator._bin(sky_out, x, sampler.every_sample())

        n_stokes = x.data.shape[0]
        x_tiled = x.data.reshape(n_stokes, tiling.n_rows, tiling.n_cols)

        def body(i: Int[Array, ''], sky: _StokesT) -> _StokesT:
            start, index, fresh = tiling.batch(i)
            size = (n_stokes, tiling.rows, tiling.cols)
            x_batch = jnp.where(fresh, lax.dynamic_slice(x_tiled, (0, *start), size), 0)
            # accumulate in place: a map per batch would cost a full-map write and add each
            return self.operator._bin(sky, type(x).from_array(x_batch), index)

        sky_out = lax.fori_loop(0, tiling.n_batches, body, sky_out)
        return sky_out


def _scaled[S: Stokes](tod: S, factor: Float[Array, '...'] | None) -> S:
    """The samples multiplied by a factor per sample, if any."""
    return tod if factor is None else type(tod).from_array(tod.data * factor)


class _Tiling(NamedTuple):
    """How the operator splits the samples into batches.

    The samples are treated as a 2D array: one row per detector (more generally, per index of all
    axes but the last) and one column per time sample (the last axis). A batch is a rectangular
    tile of `rows` x `cols` of it:

    - if a detector has fewer samples than a batch holds, a tile is several whole detectors, e.g.
      6 detectors of 20 000 samples for a batch of 2^17 = 131 072 samples;
    - otherwise it is one detector and part of its samples, e.g. 131 072 of 720 000.

    Rectangular tiles, rather than runs of consecutive samples, let the samplers compute what
    depends only on the detector, such as its quaternion, once per detector instead of once per
    sample.

    Attributes:
        shape: The shape of the samples.
        rows: Number of rows (detectors) in a tile.
        cols: Number of columns (samples of one detector) in a tile.
    """

    shape: tuple[int, ...]
    rows: int
    cols: int

    @classmethod
    def plan(cls, shape: tuple[int, ...], batch_samples: int | None) -> Self:
        n_rows, n_cols = math.prod(shape[:-1]), shape[-1]
        if batch_samples is None:
            cpu = jax.default_backend() == 'cpu'
            batch_samples = _CPU_BATCH_SAMPLES if cpu else _GPU_BATCH_SAMPLES
        if batch_samples <= 0:
            return cls(shape, n_rows, n_cols)
        if batch_samples >= n_cols:
            return cls(shape, min(batch_samples // n_cols, n_rows), n_cols)
        return cls(shape, 1, batch_samples)

    @property
    def n_rows(self) -> int:
        return math.prod(self.shape[:-1])

    @property
    def n_cols(self) -> int:
        return self.shape[-1]

    @property
    def _grid(self) -> tuple[int, int]:
        return -(-self.n_rows // self.rows), -(-self.n_cols // self.cols)

    @property
    def n_batches(self) -> int:
        return math.prod(self._grid)

    def batch(
        self, i: Int[Array, '']
    ) -> tuple[tuple[Int[Array, ''], Int[Array, '']], SampleIndex, Bool[Array, 'rows cols']]:
        """Tile `i`: its first `(row, col)`, its samples, and which of them are fresh.

        The last tile along each axis is moved back so that it ends at the last row or column
        instead of running past it, so it overlaps the tile before. Its samples in that overlap
        were already covered by the previous tile: they are not fresh, and the transpose must not
        add them a second time.
        """
        tile_row, tile_col = jnp.divmod(i, self._grid[1])
        row = jnp.minimum(tile_row * self.rows, self.n_rows - self.rows)
        col = jnp.minimum(tile_col * self.cols, self.n_cols - self.cols)
        rows = row + jnp.arange(self.rows)
        cols = col + jnp.arange(self.cols)
        outer = jnp.unravel_index(rows, self.shape[:-1]) if len(self.shape) > 1 else ()
        index = (*(axis[:, None] for axis in outer), cols[None, :])
        fresh = (rows >= tile_row * self.rows)[:, None] & (cols >= tile_col * self.cols)[None, :]
        return (row, col), index, fresh


def _rotation_angles(rotation: AbstractLinearOperator) -> Float[Array, '...']:
    """The angles of a QU rotation, or its transpose, that operator algebra may merge."""
    if isinstance(rotation, QURotationOperator) and not rotation.atomic:
        return rotation.angles
    if isinstance(rotation, QURotationTransposeOperator) and not rotation.operator.atomic:
        return -rotation.operator.angles
    raise NoReduction


class QURotationPointingRule(AbstractCompositionRule):
    """Absorb `R(theta) @ P` into `P`: the sampler rotates its samples further by `theta`."""

    left_operator_class = (QURotationOperator, QURotationTransposeOperator)
    right_operator_class = PointingOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(right, PointingOperator)
        return [right.rotated(_rotation_angles(left))]


class PointingTransposeQURotationRule(AbstractCompositionRule):
    """Absorb `P.T @ R(theta).T`, the transpose of `R(theta) @ P`, into `P.T`."""

    left_operator_class = PointingTransposeOperator
    right_operator_class = (QURotationOperator, QURotationTransposeOperator)

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, PointingTransposeOperator)
        # right is R(theta).T, i.e. R(-theta)
        return [left.operator.rotated(-_rotation_angles(right)).T]
