import pickle
from abc import abstractmethod
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from functools import cached_property, partial, reduce
from logging import Logger
from math import prod
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import lineax
import numpy as np
import pixell.enmap
import pixell.utils
from astropy.io import fits
from astropy.wcs import WCS
from fastquat import Quaternion
from jax import ShapeDtypeStruct
from jax.experimental import multihost_utils as mhu
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, DTypeLike, Float, Int64, Integer, PyTree

import furax.linalg
import furax.tree
from furax import (
    AbstractLinearOperator,
    DiagonalOperator,
    IdentityOperator,
    MaskOperator,
    OperatorTag,
    SymmetricBandToeplitzOperator,
)
from furax.core import (
    AdditionOperator,
    BlockDiagonalOperator,
    BlockSelectOperator,
    HostEnsemble,
    IndexOperator,
    map_reduce,
)
from furax.core import (
    Policy as EnginePolicy,
)
from furax.interfaces.lineax import as_lineax_operator
from furax.linalg._lanczos import LanczosWhich
from furax.obs.landscapes import (
    AstropyWCSLandscape,
    HealpixLandscape,
    StokesLandscape,
    WCSLandscape,
)
from furax.obs.operators import HWPOperator, LinearPolarizerOperator, QURotationOperator
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, StokesI, StokesIQU, StokesType, ValidStokesLiteral
from furax.profiling import format_bytes

from ._catalog import ObservationCatalog, Read, detector_table, gather_rows
from ._engine import (
    DetectorFamily,
    Family,
    StackedFamily,
    ZippedFamily,
    family_apply,
    partition,
)
from ._geometry import minimum_enclosing_arc
from ._injection import inject, load_sky, unpadded
from ._logger import logger as furax_logger
from ._model import (
    FACTORED_KINDS,
    ObservationModel,
    ObservationTemplates,
    TemplateBundle,
    acquisition_operator,
    restrict_streams,
)
from ._observation import (
    AbstractGroundObservation,
    AbstractLazyObservation,
    FileBackedLazyObservation,
    ObservationBufferShape,
    ReaderField,
)
from ._provenance import provenance, utc_now
from ._reader import ObservationReader
from ._system import (
    Batch,
    BucketModel,
    BucketNormalOperator,
    MemberOperator,
)
from .config import (
    GapTreatment,
    InjectionMode,
    LandscapeConfig,
    MapMakingConfig,
    Methods,
    NoiseSource,
    TwoLevelConfig,
    WCSConfig,
    WeightingMode,
)
from .gap_filling import gap_fill
from .layout import Bucket, SlotLayout, SlotPlacement
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel
from .preconditioner import BJPreconditioner, make_two_level_preconditioner
from .results import MapMakingResults
from .templates.gram import factor
from .templates.pomme import PommeProjectionOperator
from .weight import WeightOperator


@register_dataclass
@dataclass(frozen=True)
class _AccumulatedModel:
    """Per-bucket observation models and the map-domain sums."""

    buckets: tuple[BucketModel, ...]
    """One `BucketModel` per bucket of the layout, in bucket order."""
    hit_map: Int64[Array, '...']
    """Hit map over every observation, replicated on every device."""
    map_rhs: StokesType
    """Map RHS over every observation, replicated on every device."""


class _MapMakingSystem(NamedTuple):
    """The normal system handed to the solver."""

    A: AbstractLinearOperator
    """The normal operator, acting on the selected sky pixels (plus amplitudes, if any)."""
    b: PyTree
    """The right-hand side, in the same unknowns."""
    preconditioner: AbstractLinearOperator
    """Approximate inverse of `A`."""
    has_amplitudes: bool
    """Whether the unknowns include explicit-template amplitudes."""
    response_b: PyTree | None = None
    """The right-hand side of the simulated sky, in the same unknowns, if it is injected as a
    response."""


class _Solution(NamedTuple):
    """The solution of the normal system, and what comes with it."""

    sky: PyTree[Array]
    """The selected sky pixels."""
    amplitudes: dict[str, np.ndarray] | None
    """The explicit-template amplitudes per observation, if the system has some."""
    injected: PyTree[Array] | None
    """The selected sky pixels of the injected sky's solution, if one is solved for."""
    stats: dict[str, Any]
    """The solver statistics."""


class MultiObservationMapMaker[T]:
    """Class for mapping multiple observations together."""

    def __init__(
        self,
        observations: Sequence[AbstractLazyObservation[T]],
        config: MapMakingConfig | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.observations = observations
        self.config = config or MapMakingConfig()  # use defaults if not provided
        self.logger = logger or furax_logger
        self._check_config()
        self.landscape = (
            _static_landscape(self.config.landscape, self.config.dtype)
            or self._scan_wcs_footprint()
        )
        self.sky = self._load_sky()
        # Build the readers once, up front: they only need the probe shapes (no mesh context)
        rhs_fields = {ReaderField.METADATA, ReaderField.SAMPLE_DATA}
        model_fields = ObservationModel.required_reader_fields(self.config)
        template_fields = ObservationTemplates.required_reader_fields(self.config)
        self.reader_fields = model_fields | rhs_fields | template_fields
        self.readers = self.get_readers(self.reader_fields)

    def _load_sky(self) -> StokesType | None:
        """The simulated sky, if one is injected.

        A sky scanned into the TOD has the Stokes components that are read; one solved for as a
        response, those of the map.
        """
        if (injection := self.config.injection) is None:
            return None
        if injection.mode == InjectionMode.RESPONSE:
            return load_sky(injection, self.landscape)
        return load_sky(injection, self.landscape.with_stokes(self.config.read_stokes))

    @property
    def _scanned_sky(self) -> StokesType | None:
        """The simulated sky scanned into the TOD, if any."""
        injection = self.config.injection
        if injection is None or injection.mode == InjectionMode.RESPONSE:
            return None
        return self.sky

    def _check_config(self) -> None:
        """Validate and adjust config for method-specific compatibility."""
        if self.config.method == Methods.POMME:
            if not self.config.binned:
                raise ValueError('Pomme requires diagonal weighting (weighting.mode=DIAGONAL).')
            if 'I' in (stokes := self.config.landscape.stokes):
                if stokes != 'IQU':
                    raise ValueError(
                        f'Pomme does not support intensity map reconstruction and {stokes=!r}'
                        " cannot be reduced to a supported type. Use stokes='QU' instead."
                    )
                self.logger.info(
                    "Received stokes='IQU', but Pomme does not support intensity map"
                    " reconstruction. Falling back to stokes='QU' instead."
                )
                self.config.landscape.stokes = 'QU'
            if self.config.demodulated:
                raise ValueError('Pomme acts on the HWP-modulated TOD, not on demodulated data.')
        if (
            self.config.use_templates
            and not self.config.binned
            and self.config.gaps.treatment != GapTreatment.INNER_MASK
        ):
            # gap filling bypasses the template path, and the nested weight couples samples
            # across whole gaps, beyond the band of the template Grams
            raise NotImplementedError(
                'Templates with correlated weighting require gaps.treatment=inner_mask.'
            )

    @property
    def n_observations(self) -> int:
        """Total number of observations across all processes."""
        return len(self.observations)

    @cached_property
    def _catalog(self) -> ObservationCatalog:
        """What the run knows of every observation, starting from its probe.

        Probing costs an open, so the observations are dealt round-robin to the processes and the
        probes are gathered. A probe that raises must not crash the process (it would deadlock the
        others at the gather), so its exception is kept instead.
        """
        n_proc = jax.process_count()
        need_intervals = ReaderField.SCANNING_INTERVALS in self.reader_fields
        local: dict[int, ObservationBufferShape | str] = {}
        for i in range(jax.process_index(), self.n_observations, n_proc):
            try:
                local[i] = self.observations[i].probe_shape(need_intervals)
            except Exception as error:
                self.logger.exception('probe of observation %d failed', i)
                local[i] = f'{type(error).__name__}: {error}'
        names = [observation.name for observation in self.observations]
        return ObservationCatalog(names, gather_rows(local))

    @cached_property
    def _probe_shapes(self) -> list[ObservationBufferShape]:
        """Every observation's buffer shape.

        Observations whose probe failed are skipped by the readers and gated out of the
        accumulation. They inherit the largest successful probe shape so they cannot form an
        undersized bucket of their own.

        The sample count is raised to `_minimum_buffer_samples`, and the detector count to whole
        detector batches, so the buckets and the readers size their buffers identically.
        """
        rows = self._catalog.probed
        failed = self._catalog.probe_failed
        rows[failed] = rows[~failed].max(axis=0) if np.any(~failed) else (1, 1, 0)
        minimum_samples = self._minimum_buffer_samples
        return [
            ObservationBufferShape(
                int(row[0]), max(int(row[1]), minimum_samples), int(row[2])
            ).whole_detector_batches(self.config.detector_batch_size)
            for row in rows
        ]

    @property
    def _minimum_buffer_samples(self) -> int:
        """Smallest sample envelope accepted by the configured TOD operators."""
        minimum = self.config.pomme_tau if self.config.method == Methods.POMME else 1
        if self.config.weighting.mode == WeightingMode.TOEPLITZ:
            correlation_length = self.config.weighting.correlation_length
            if self.config.gaps.treatment == GapTreatment.FILL:
                # The constrained realization embeds the symmetric band in a circulant kernel.
                correlation_length = 2 * correlation_length - 1
            minimum = max(minimum, correlation_length)
        return minimum

    @cached_property
    def layout(self) -> SlotLayout:
        """How the observations are bucketed and laid out over the devices."""
        return SlotLayout.create(
            self._probe_shapes,
            n_devices=jax.device_count(),
            max_buckets=self.config.max_buckets,
            detector_batch_size=self.config.detector_batch_size,
        )

    @cached_property
    def placement(self) -> SlotPlacement:
        """The slots of the layout, placed on the devices of the job."""
        return SlotPlacement.create(self.layout)

    @cached_property
    def fit_config(self) -> MapMakingConfig:
        """The run configuration, with the Welch segment clipped to the shortest observation.

        [`padding_aware_welch`][furax.mapmaking.noise.padding_aware_welch] drops the segments that
        fall in an observation's padded tail, but the segment length sets the number of frequency
        bins and so cannot vary with the padding. A segment longer than the shortest observation
        would therefore leave one partial segment straddling that observation's padding, and its
        noise fit would see the zeros. Clipping the segment keeps every averaged segment inside
        real data.

        The clip is taken over the whole dataset rather than over each bucket, so that the noise
        fit stays independent of how the observations happen to be bucketed. Observation lengths
        come from the probe, which is an upper bound: an observation that reads shorter than probed
        still falls back to a partial segment.
        """
        weighting = self.config.weighting
        if weighting.mode == WeightingMode.IDENTITY or weighting.source != NoiseSource.FIT:
            return self.config  # no PSD is estimated, so the segment length is unused
        shortest = min(shape.sample_count for shape in self._probe_shapes)
        nperseg = min(weighting.fitting.nperseg, shortest)
        if nperseg == weighting.fitting.nperseg:
            return self.config
        self.logger.info(
            f'MultiObsMapMaker: Welch segment clipped to {nperseg} samples '
            f'(configured {weighting.fitting.nperseg}) by the shortest observation'
        )
        fitting = replace(weighting.fitting, nperseg=nperseg)
        return replace(self.config, weighting=replace(weighting, fitting=fitting))

    def get_readers(self, required_fields: Collection[str]) -> tuple[ObservationReader[T], ...]:
        """Build one reader per bucket, over that bucket's observations only.

        A reader reads its bucket's observations in slot order, so its buffers are sized by
        exactly the observations it can be asked to read.
        """
        shapes = self._probe_shapes
        failed = self._catalog.probe_failed
        return tuple(
            ObservationReader.from_observations(
                [self.observations[i] for i in bucket.observations],
                requested_fields=required_fields,
                stokes=self.config.read_stokes if self.config.demodulated else None,
                dtype=self.config.dtype,
                sample_dtype=jnp.float32,
                shapes=[shapes[i] for i in bucket.observations],
                known_failures=np.flatnonzero(failed[bucket.observations]).tolist(),
            )
            for bucket in self.layout.buckets
        )

    def run(self, out_dir: str | Path | None = None) -> MapMakingResults:
        """Runs the mapmaker and return results after saving them to the given directory."""
        results = self.make_maps()

        # Save outputs on process 0 only (all processes hold the same maps; only process 0 holds
        # the provenance)
        if out_dir is not None and jax.process_index() == 0:
            out_dir = Path(out_dir).resolve()
            results.save(out_dir)
            self.logger.info(f'saved results to {out_dir}')
            self.config.dump_yaml(out_dir / 'mapmaking_config.yaml')
            self.logger.info('saved mapmaking configuration to file')

        # Barrier so other ranks don't race ahead while rank 0 is still writing.
        mhu.sync_global_devices('mapmaker.run.save_done')

        return results

    def _log_layout(self) -> None:
        """Log the device layout and what the bucketing costs in padding (rank 0 only)."""
        if jax.process_index() != 0:
            return
        layout = self.layout
        self._info(
            f'layout procs={jax.process_count()} dev_per_proc={len(self.placement.local_devices)} '
            f'dev_total={layout.n_devices}'
        )
        # Every slot of a bucket is padded to its reader's out_structure, real or empty.
        real_bytes = sum(reader.total_nbytes for reader in self.readers)
        padded_bytes = 0
        for b, (bucket, reader) in enumerate(zip(layout.buckets, self.readers, strict=True)):
            slot_bytes = furax.tree.nbytes(reader.out_structure)
            padded_bytes += slot_bytes * bucket.n_slots
            self._info(
                f'bucket {b}: {bucket.summary(layout.n_devices)} '
                f'slot_size={format_bytes(slot_bytes)} '
                f'real_size={format_bytes(reader.total_nbytes)} '
                f'pad_size={format_bytes(slot_bytes * bucket.n_slots - reader.total_nbytes)}'
            )
        byte_overhead = (padded_bytes - real_bytes) / real_bytes
        self._info(
            f'dataset {layout.summary()} real={format_bytes(real_bytes)} '
            f'padded={format_bytes(padded_bytes)} byte_overhead=+{byte_overhead:.1%}'
        )

    def make_maps(self) -> MapMakingResults:
        """Computes the mapmaker results (maps and other products)."""
        rank = jax.process_index()
        started = utc_now()
        self._log_layout()

        with jax.set_mesh(self.placement.mesh):
            # Single read pass, so each observation is read/preprocessed exactly once.
            acc = self.build_model_and_accumulate()
            jax.block_until_ready(acc)
            self._info('Accumulated hit map and RHS vector')

            self._catalog.reads = self._gather_reads()
            if n_failed := self._catalog.n_failed:
                self._info(f'{n_failed} observation(s) failed and were excluded')
            noise_parameters = None
            if self.config.weighting.mode != WeightingMode.IDENTITY:
                noise_parameters = self._collect_noise_parameters(acc)

            BJ = self._block_jacobi(acc)
            icov = BJ.blocks.block_until_ready()
            self._info('Computed white noise inverse covariance')

            valid_pixels = self.pixel_selection(acc.hit_map, icov)
            # Select valid pixels on the (trailing) sky axes, leaving the leading Stokes axis intact.
            S = IndexOperator(
                (..., *jnp.where(valid_pixels)), in_structure=self.landscape.structure
            )
            n_observed = jnp.sum(acc.hit_map > 0)
            self._info(
                f'Selected {jnp.sum(valid_pixels)} pixels ({n_observed} seen, '
                f'{valid_pixels.size} total)'
            )

            response_sky = None if self._scanned_sky is not None else self.sky
            system = self._build_system(acc, S, BJ, response_sky)
            if (two_level := self.config.solver.two_level) is not None:
                system = self._deflate(system, two_level)
            solution = self._solve(system, acc)

        # Built only now, so that their copies are not held through the solve
        return MapMakingResults(
            map=S.T(solution.sky),  # all sky pixels including those not estimated (zero)
            icov=jnp.moveaxis(icov, [-2, -1], [0, 1]),  # (*pixels, ns, ns) → (ns, ns, *pixels)
            hit_map=jnp.where(valid_pixels, acc.hit_map, 0),  # excluded pixels have zero hits
            solver_stats=solution.stats,
            landscape=self.landscape,
            template_amplitudes=solution.amplitudes,
            injected_map=None if solution.injected is None else S.T(solution.injected),
            observations=self._catalog.to_table(),
            # the noise model covers the streams of the model's TOD, not all those read
            detectors=detector_table(
                self._catalog.reads,
                noise_parameters,
                acc.buckets[0].batches.model.tod_format.stokes,
            ),
            # only process 0 saves it, so only process 0 spends the git calls on it
            provenance=provenance(started=started) if rank == 0 else None,
        )

    def _info(self, message: str) -> None:
        self.logger.info(f'MultiObsMapMaker: {message}')

    def _block_jacobi(self, acc: _AccumulatedModel) -> BJPreconditioner:
        """The block-Jacobi preconditioner, from the diagonal pixel system summed over the buckets.

        The system is under the weight without the noise correlations, nor the implicit templates.
        """
        weight = 'plain' if self.config.binned else 'white'
        A_diag = AdditionOperator(
            [
                BucketNormalOperator(
                    bm,
                    None,
                    self.placement.axis,
                    weight,
                    amplitudes=False,
                    in_structure=self.landscape.structure,
                )
                for bm in acc.buckets
            ],
            sequential=True,
        )
        return BJPreconditioner.create(A_diag)

    def _solve(self, system: _MapMakingSystem, acc: _AccumulatedModel) -> _Solution:
        """The solution of the system, and of the injected sky's, if any, and their statistics."""
        rank = jax.process_index()

        def log_iteration(step: Array, r_norm: Array) -> None:
            if rank == 0:  # log from rank 0 only
                self._info(f'CG step={int(step)} residual={float(r_norm):.6e}')

        def solve(b):
            return furax.linalg.cg(
                system.A,
                b,
                preconditioner=system.preconditioner,
                iteration_callback=log_iteration if self.config.solver.verbose else None,
                **self.config.solver.options,
            )

        result = solve(system.b)
        self._info(f'Finished GLS solve ({int(result.num_steps)} it)')
        stats = _solver_stats(result, self.config.solver.options)
        injected = None
        if system.response_b is not None:
            # the same normal system, with the simulated sky as the right-hand side
            response = solve(system.response_b)
            self._info(f'Finished the solve of the injected sky ({int(response.num_steps)} it)')
            stats['injected'] = _solver_stats(response, self.config.solver.options)
            injected = response.solution if not system.has_amplitudes else response.solution[0]

        if not system.has_amplitudes:
            return _Solution(result.solution, None, injected, stats)
        sky, *blocks = result.solution
        return _Solution(sky, self._template_amplitudes(acc, blocks), injected, stats)

    def _template_amplitudes(
        self, acc: _AccumulatedModel, blocks: list[list[PyTree[Array]]]
    ) -> dict[str, np.ndarray]:
        """The explicit-template amplitudes solved for, per observation, from their CG blocks.

        Every device holds the same replicated copy after the gather, so the host array is
        complete on every process; then back to observation order.
        """
        # Explicit templates are configured for the run, so every bucket carries them or none.
        templates = acc.buckets[0].templates
        explicit = None if templates is None else templates.explicit
        amplitudes = {}
        if explicit is not None:
            own, *blocks = blocks
            per_bucket = [self.placement.gather(a) for a in own]
            for name in per_bucket[0]:
                # the detector axis follows the stream axis, if any (see `Template`), after the
                # slot and batch axes
                axis = 2 + explicit.operator.templates[name].per.index('det')
                amplitudes[name] = self.layout.to_observation_order(
                    [a[name] for a in per_bucket], detector_axis=axis
                )
        for common in blocks:  # one set per slot, no detector axis
            amplitudes |= jax.tree.map(
                lambda *leaves: self.layout.to_observation_order(leaves),
                *[self.placement.gather(a) for a in common],
            )
        return amplitudes

    def _build_system(
        self,
        acc: _AccumulatedModel,
        S: AbstractLinearOperator,
        BJ: BJPreconditioner,
        sky: StokesType | None = None,
    ) -> _MapMakingSystem:
        r"""Assembles the GLS normal system $J^T W' J x = J^T W' d$ over the selected pixels.

        Args:
            acc: The accumulated per-bucket models, hit map and map RHS.
            S: Selection of the estimated pixels out of the full sky grid.
            BJ: Block-Jacobi preconditioner.
            sky: A simulated sky on the full grid, whose right-hand side `response_b` is added.

        Returns:
            The system to solve, in the unknowns described above.
        """
        axis = self.placement.axis
        # Restrict BJ preconditioner to selected pixels
        M = (S @ BJ.I @ S.T).reduce()

        # Explicit templates are configured for the run, so every bucket carries them or none.
        templates = acc.buckets[0].templates
        kinds = []
        if templates is not None:
            kinds = [kind for kind in FACTORED_KINDS if getattr(templates, kind) is not None]

        if not kinds:
            terms = [
                BucketNormalOperator(bm, S, axis, in_structure=S.out_structure)
                for bm in acc.buckets
            ]
            A = AdditionOperator(terms, sequential=True)
            response = None
            if sky is not None:
                response = reduce(furax.tree.add, [op.response(sky) for op in terms])
            return _MapMakingSystem(A, S(acc.map_rhs), M, has_amplitudes=False, response_b=response)

        # The unknowns are the selected sky pixels, then, per kind of amplitudes present, one block
        # per bucket: those of each detector batch alone, `(slot, batch, ...)`, and those the
        # detectors of an observation share, `(slot, ...)`.
        structures = {
            kind: [
                self._amplitude_structure(bucket, bm, kind)
                for bucket, bm in zip(self.layout.buckets, acc.buckets, strict=True)
            ]
            for kind in kinds
        }
        at = {kind: 1 + k for k, kind in enumerate(kinds)}  # their block among the unknowns

        # Joint system. Each bucket's system sees the selected pixels and its own amplitudes,
        # picked out of the lists and embedded back by `E`.
        terms, responses = [], []
        for b, bm in enumerate(acc.buckets):
            in_structure = [S.out_structure, *(structures[kind][b] for kind in kinds)]
            normal = BucketNormalOperator(bm, S, axis, in_structure=in_structure)
            picks = [BlockSelectOperator(b, in_structure=structures[kind]) for kind in kinds]
            E = BlockDiagonalOperator([IdentityOperator(in_structure=S.out_structure), *picks])
            terms.append((E.T @ normal @ E).reduce())
            if sky is not None:
                responses.append(E.T(normal.response(sky)))

        # The Gram inverses of the amplitudes of each detector batch alone are per batch, those
        # of the common amplitudes per observation.
        inverses: dict[str, list[AbstractLinearOperator]] = {kind: [] for kind in kinds}
        ridges: dict[str, list[AbstractLinearOperator | None]] = {kind: [] for kind in kinds}
        for b, bm in enumerate(acc.buckets):
            for kind in kinds:
                inverse, ridge = self._gram_inverse(bm, kind, structures[kind][b])
                inverses[kind].append(inverse)
                ridges[kind].append(ridge)

        # The ridge the Gram inverses were factored with acts on the amplitudes alone, so it is
        # one term over them all, outside the passes over the TOD.
        for kind in kinds:
            ridged = ridges[kind]
            assert all(r is None for r in ridged) or all(r is not None for r in ridged)
            if ridged[0] is not None:
                amplitudes = BlockSelectOperator(at[kind], in_structure=terms[0].in_structure)
                terms.append((amplitudes.T @ BlockDiagonalOperator(ridged) @ amplitudes).reduce())

        rhs = [
            S(acc.map_rhs),
            *(
                # declared stacked over the batches, the right-hand sides of each batch's own
                # amplitudes are wholly per batch
                [
                    bm.batches.amplitude_rhs if FACTORED_KINDS[kind] == 'stack' else bm.common_rhs
                    for bm in acc.buckets
                ]
                for kind in kinds
            ),
        ]
        M = BlockDiagonalOperator([M, *(inverses[kind] for kind in kinds)])
        A = AdditionOperator(terms, sequential=True)
        response = None
        if sky is not None:
            response = reduce(furax.tree.add, responses)
        return _MapMakingSystem(A, rhs, M, has_amplitudes=True, response_b=response)

    @staticmethod
    def _amplitude_structure(
        bucket: Bucket, bm: BucketModel, kind: str
    ) -> dict[str, jax.ShapeDtypeStruct]:
        """The amplitudes of a kind of a bucket.

        Those of each batch alone are `(slot, batch, ...)`, those of each observation `(slot, ...)`.
        """
        assert bm.templates is not None
        bundle = getattr(bm.templates, kind)
        assert bundle is not None
        leading = (bucket.n_slots,)
        if FACTORED_KINDS[kind] == 'stack':
            leading += (bucket.n_batches,)
        return {
            name: jax.ShapeDtypeStruct((*leading, *s.shape), s.dtype)
            for name, s in bundle.operator.in_structure.items()
        }

    @staticmethod
    def _gram_inverse(
        bm: BucketModel, kind: str, structure: dict[str, jax.ShapeDtypeStruct]
    ) -> tuple[AbstractLinearOperator, AbstractLinearOperator | None]:
        """The inverse of the Gram of a kind of amplitudes of a bucket, and the ridge added."""

        def bundle(batch: Batch) -> TemplateBundle:
            assert batch.templates is not None
            bundle = getattr(batch.templates, kind)
            assert bundle is not None
            return bundle

        # those of each batch alone hold per batch what depends on the batch, the others whole
        own, kept = bundle(bm.batches), bundle(bm.shared)

        def member(part: str) -> AbstractLinearOperator | None:
            operators, shared = getattr(own, part), getattr(kept, part)
            if operators is None and shared is None:
                return None
            if FACTORED_KINDS[kind] == 'stack':
                return MemberOperator(operators, shared, batches=True, in_structure=structure)
            return MemberOperator(shared, in_structure=structure)

        inverse = member('gram_inverse')
        assert inverse is not None
        return inverse, member('ridge')

    def _deflate(self, system: _MapMakingSystem, config: TwoLevelConfig) -> _MapMakingSystem:
        """Replaces the system's preconditioner by a two-level one deflating some of its modes."""
        preconditioned = config.spectrum == 'preconditioned'
        # The spectrum is positive, so its smallest and largest eigenvalues are its algebraic ends.
        lanczos_which: dict[str, LanczosWhich] = {'small': 'SA', 'large': 'LA', 'both': 'BE'}
        result = furax.linalg.lanczos_tr(
            system.A,
            key=jax.random.key(config.seed),
            k=config.rank,
            m=config.krylov_size,
            which=lanczos_which[config.which],
            max_restarts=config.max_restarts,
            tol=config.tol,
            preconditioner=system.preconditioner if preconditioned else None,
        )
        theta = result.eigenvalues
        self._info(
            f'Deflating {config.rank} eigenpairs ({config.which}) of the '
            f'{config.spectrum} spectrum, eigenvalues in '
            f'[{float(theta[0]):.6e}, {float(theta[-1]):.6e}], '
            f'max residual bound {float(jnp.max(result.residual_norms)):.3e}'
        )
        M = make_two_level_preconditioner(
            system.preconditioner,
            furax.linalg.LowRankTerms(theta, result.eigenvectors),
            preconditioned=preconditioned,
        )
        return system._replace(preconditioner=M)

    def _gather_reads(self) -> dict[int, Read]:
        """What the last pass read of each observation, or the exception that stopped its load.

        Keyed by observation index, gathered from the readers of every process. An observation
        that was not loaded, such as one whose probe failed, is absent.
        """
        local: dict[int, Read] = {}
        for bucket, reader in zip(self.layout.buckets, self.readers, strict=True):
            for slot, record in reader.records.items():
                local[int(bucket.observations[slot])] = record
            for slot, error in reader.errors.items():
                local[int(bucket.observations[slot])] = error
        return gather_rows(local)

    def _collect_noise_parameters(self, acc: _AccumulatedModel) -> dict[str, np.ndarray]:
        """The noise model parameters of every detector slot, in observation order.

        Each parameter is `(n_observations, [stream,] detector)`, its detector axis padded.
        """
        per_bucket = []
        for bm in acc.buckets:
            # every parameter is a detector's, so held per batch: `(slot, batch, ..., detector)`
            leaves, _ = jax.tree_util.tree_flatten_with_path(
                self.placement.gather(bm.batches.model.noise_model)
            )
            per_bucket.append({_parameter_name(path): leaf for path, leaf in leaves})
        return {
            name: self.layout.to_observation_order([p[name] for p in per_bucket], detector_axis=-1)
            for name in per_bucket[0]
        }

    def build_model_and_accumulate(self) -> _AccumulatedModel:
        """Build the model and accumulate the hit map and map RHS in one pass over the data.

        Empty (or failed) slots are gated out using a zero-mask, so they drop from the system.

        This method must run under `jax.set_mesh(self.placement.mesh)`.
        """
        hit_map = jnp.zeros(self.landscape.shape, jnp.int64)
        map_rhs = self.landscape.zeros()
        # The sky to scan in is an argument of the kernels, held by every device for this pass
        # only, rather than a constant compiled into them
        sky = None
        if (scanned := self._scanned_sky) is not None:
            sky = self.placement.broadcast(scanned)
        bucket_models = []
        for b, reader in enumerate(self.readers):
            hits, rhs, bucket_model = self._accumulate_bucket(b, reader, sky)
            hit_map = hit_map + hits
            map_rhs = furax.tree.add(map_rhs, rhs)
            bucket_models.append(bucket_model)
        return _AccumulatedModel(buckets=tuple(bucket_models), hit_map=hit_map, map_rhs=map_rhs)

    def _accumulate_bucket(
        self, bucket_index: int, reader: ObservationReader[T], sky: StokesType | None
    ) -> tuple[Int64[Array, '...'], StokesType, BucketModel]:
        """One bucket's pass over the data: its hit map and RHS partials, and its stacked model.

        The pass is an engine call on a host ensemble of the bucket's slots: each round launches
        one observation on each device while its process reads the next round, so only one
        observation per device is in memory at a time, and the processes need not keep in step
        until the device sum at the end. The stacked outputs land sharded over the slots, in slot
        order, which is the layout `BucketModel` holds.
        """
        reader.reset_failures()  # fresh pass: drop failures recorded by any previous read
        bucket = self.layout.buckets[bucket_index]

        def read(slot: int) -> tuple[np.ndarray, ...]:
            if not bucket.is_empty[slot]:
                member = reader.read_host(slot)  # readers are indexed by slot
            else:
                # An empty slot is never read or preprocessed just to be masked away
                member = reader.read_filler_host()
            return tuple(np.asarray(leaf) for leaf in jax.tree.leaves(member))

        template = (
            reader.out_structure,
            jax.tree.map(
                lambda s: jax.ShapeDtypeStruct((len(s.shape),), np.int32), reader.out_structure
            ),
            jax.ShapeDtypeStruct((), bool),
        )
        ensemble = HostEnsemble(
            bucket.n_slots,
            read,
            template,
            policy=EnginePolicy(axis=self.placement.axis),
            prefetch=self._prefetches_reads,
        )
        (hits, rhs, out), _ = map_reduce(
            self._observation_kernel(bucket), ensemble, ('sum', 'sum', 1), sky
        )
        return hits, rhs, BucketModel(*out)

    @cached_property
    def _prefetches_reads(self) -> bool:
        """Whether to read the next observations while the devices accumulate the current ones.

        Only where the devices have memory of their own: on a CPU device, the reads would hold the
        next TOD in the same memory as the current one's working set, and compete with it for
        the cores.
        """
        return self.placement.local_devices[0].platform != 'cpu'

    def _observation_kernel(
        self, bucket: Bucket
    ) -> Callable[[tuple[Any, Any, Array], StokesType | None], tuple[Array, StokesType, Any]]:
        """Accumulate one observation: its hit map, RHS and model.

        The member is one slot's read, the triple `(observation, padding, valid)`; the sky to
        inject, if any, is a shared input that every member receives whole.
        """
        config = self.fit_config
        landscape = self.landscape
        fill_gaps = config.gaps.treatment == GapTreatment.FILL and not config.binned
        build_templates = config.use_templates
        n_batches, batch_size = bucket.n_batches, bucket.batch_size
        sky_landscape = landscape.with_stokes(config.read_stokes)
        replaces = config.injection is not None and config.injection.mode == InjectionMode.REPLACE
        regularization = 0.0 if config.templates is None else config.templates.regularization
        factored = partial(factor, regularization=regularization)

        def cast(tod):
            # The TOD is read in float32, whatever the pipeline's dtype
            return jax.tree.map(lambda x: x.astype(config.dtype), tod)

        def cast_tod(data):
            return {**data, ReaderField.SAMPLE_DATA: cast(data[ReaderField.SAMPLE_DATA])}

        def kernel(member, sky):
            observation, padding, use = member
            injection = None
            if sky is not None:
                shape = jax.tree.leaves(observation[ReaderField.SAMPLE_DATA])[0].shape
                injection = (sky, unpadded(padding, shape))
            return step(observation, padding, use, injection)

        def scan_in(data, injection):
            if injection is None:
                return data
            sky, injectable = injection
            H = acquisition_operator(data, config, sky_landscape)
            return inject(H, sky, data, injectable, replaces)

        def step(observation, padding, use, injection):
            # One detector batch at a time: nothing computed from the TOD outlives its batch, each
            # batch contributing to the sums and stacking its own model and templates. Gap filling
            # needs whole observations, so it only runs with a single batch.
            assert not fill_gaps or n_batches == 1
            family: Family = DetectorFamily(observation, batch_size)
            if injection is not None:  # the samples to inject into, per detector
                injectable = StackedFamily(injection[1], axis=-2, width=batch_size)
                family = ZippedFamily((family, injectable))

            def apply(members):
                if injection is None:
                    data = cast_tod(members)
                else:
                    data, injectable = members
                    data = scan_in(cast_tod(data), (injection[0], injectable))
                tod = raw = data[ReaderField.SAMPLE_DATA]
                if fill_gaps:
                    # The RHS uses the filled TOD; the models are still built from the raw one.
                    tod = jax.lax.cond(
                        use,
                        lambda: fill_observation_gaps(data, padding),
                        lambda: raw,  # nothing to fill
                    )
                return accumulate(data, tod, padding, use)

            # The sums run over the batches; of the rest, what the batches share (boresight
            # pointing, common template bases, ...) is kept once per observation. The Gram
            # inverses and right-hand sides of the templates go where their amplitudes live.
            # A Gram summed over the batches is paired with its coarse amplitudes, which follow
            # from the bases alone, the same in every batch.
            spec = {
                **dict.fromkeys(('hits', 'map'), 'sum'),
                **dict.fromkeys(('model', 'implicit', 'operators'), 'auto'),
                'inverses': 'stack',
                'grams': {
                    kind: ('sum', 'shared')
                    for kind, held in FACTORED_KINDS.items()
                    if held == 'sum'
                },
                'rhs': FACTORED_KINDS,
            }
            out, kinds = family_apply(family, apply, spec)
            templates = template_kinds = None
            if build_templates:
                inverses = out['inverses'] | {
                    kind: None if gram is None else factored(replace(gram, coarse=coarse))
                    for kind, (gram, coarse) in out['grams'].items()
                }
                templates = ObservationTemplates.from_inverses(
                    out['implicit'], out['operators'], inverses
                )
                # an inverse factored once per observation is held whole by it
                inverse_kinds = kinds['inverses'] | {
                    kind: jax.tree.map(lambda _: 'shared', inverses[kind]) for kind in out['grams']
                }
                template_kinds = ObservationTemplates.from_inverses(
                    kinds['implicit'], kinds['operators'], inverse_kinds
                )
            batch = Batch(out['model'], templates, out['rhs']['explicit'])
            batch_kinds = Batch(kinds['model'], template_kinds, kinds['rhs']['explicit'])
            batches, kept = partition(batch, batch_kinds)
            return out['hits'], out['map'], (batches, kept, out['rhs']['common'])

        def fill_observation_gaps(data, padding):
            obs = ObservationModel.create(data, padding, config, landscape)
            # Only reached under GapTreatment.FILL, where W is the plain inner-mask weight.
            assert isinstance(obs.W, WeightOperator)
            # Optional M_b N M_b preconditioner (covariance from the noise model).
            preconditioner = None
            if config.gaps.fill_options.precondition:
                cov = obs.noise_operator(config.weighting.correlation_length, inverse=False)
                m_bad = obs.M.complement()
                preconditioner = (m_bad @ cov @ m_bad).reduce()
            return gap_fill(
                jax.random.key(config.gaps.fill_options.seed),
                data[ReaderField.SAMPLE_DATA],
                obs.W.weight,
                obs.M,
                rate=obs.sample_rate,
                max_cg_steps=config.gaps.fill_options.max_steps,
                rtol=config.gaps.fill_options.rtol,
                preconditioner=preconditioner,
                metadata=data[ReaderField.METADATA],
            )

        def accumulate(data, tod, padding, use):
            temperature = None
            if config.read_stokes != landscape.stokes:
                # The extra streams read, the I one for the T2P template, stay out of the model
                temperature = data[ReaderField.SAMPLE_DATA].i
                data, tod = restrict_streams(data, tod, landscape.stokes)
            obs = ObservationModel.create(data, padding, config, landscape)

            # Padding/failed observations contribute nothing
            obs.M = obs.M.restrict(use)

            # Hit map = nearest-neighbour coverage of the sample mask
            hit_pointing = PointingOperator.create(
                landscape,
                Quaternion.from_array(data[ReaderField.BORESIGHT_QUATERNIONS]),
                Quaternion.from_array(data[ReaderField.DETECTOR_QUATERNIONS]),
            ).as_stokes_i(interpolate=False)
            # Read the mask directly: M(ones) = M.to_boolean_mask()
            masked_tod = obs.M.to_boolean_mask()
            # The mask is (ndet, nsamp) even in the demodulated case (all streams share the same)
            masked = masked_tod.data if isinstance(masked_tod, Stokes) else masked_tod
            hits_i = jnp.int64(hit_pointing.T(StokesI(masked)).i)

            if not build_templates:
                if fill_gaps:
                    # Gaps filled: skip the data-side mask so the fill survives N⁻¹.
                    rhs_i = obs.rhs_operator_prefilled(tod)
                else:
                    rhs_i = obs.rhs_operator(tod)
                absent = dict.fromkeys(FACTORED_KINDS)
                return {
                    'hits': hits_i,
                    'map': rhs_i,
                    'model': obs,
                    'implicit': None,
                    'operators': absent,
                    'inverses': {kind: None for kind in absent if FACTORED_KINDS[kind] == 'stack'},
                    'grams': {
                        kind: (None, None) for kind in absent if FACTORED_KINDS[kind] == 'sum'
                    },
                    'rhs': absent,
                }

            # Templates are rejected with gap filling (see `_check_config`).
            built = ObservationTemplates.assemble(data, config, obs, tod, temperature)
            wd = built.weighted_tod
            return {
                'hits': hits_i,
                'map': obs.H.T(wd),
                'model': obs,
                'implicit': built.implicit,
                'operators': built.operators,
                'inverses': {
                    kind: None if gram is None else factored(gram)
                    for kind, gram in built.grams.items()
                    if FACTORED_KINDS[kind] == 'stack'
                },
                'grams': {
                    kind: (None, None)
                    if gram is None
                    else (replace(gram, coarse=None), gram.coarse)
                    for kind, gram in built.grams.items()
                    if FACTORED_KINDS[kind] == 'sum'
                },
                # the batch's share of the right-hand side of each kind of amplitudes
                'rhs': {
                    kind: None if op is None else op.T(wd) for kind, op in built.operators.items()
                },
            }

        return kernel

    def pixel_selection(
        self, hits: Integer[Array, ' pixels'], weights: Float[Array, 'pixels stokes stokes']
    ) -> Bool[Array, ' pixels']:
        """Compute pixel selection according to hit and condition number cuts."""
        # Cut pixels with low number of samples
        hits_quantile = jnp.quantile(hits[hits > 0], q=0.95)
        valid = hits > self.config.hits_cut * hits_quantile

        if self.config.cond_cut > 0:
            eigs = furax.linalg.eigvalsh(weights)
            valid = jnp.logical_and(
                valid,
                eigs[..., 0] > self.config.cond_cut * eigs[..., -1],
            )

        return valid

    def _scan_wcs_footprint(self) -> WCSLandscape:
        """Scan observations to determine the combined WCS footprint and build a WCSLandscape.

        Performs a preliminary pass over all observations, loading the pointing data needed
        to compute each observation's sky coverage, then combines them into a unified footprint.

        Restricted to single-process runs over file-backed observations: the scan is an
        unsharded loop calling ``get_data`` per observation. For file-backed observations that
        cheaply loads only the requested pointing fields, but for sources that cannot subset
        their fields (e.g. preproc-backed observations) it would run the full load/preprocess
        pipeline on every observation, on every process. Pre-compute the footprint and pass an
        explicit WCS landscape instead in those cases.
        """
        if jax.process_count() > 1:
            msg = (
                'Automatic WCS footprint scanning is only supported on a single process. '
                'Pre-compute the footprint and pass an explicit WCS landscape instead.'
            )
            raise RuntimeError(msg)
        if any(not isinstance(obs, FileBackedLazyObservation) for obs in self.observations):
            msg = (
                'Automatic WCS footprint scanning requires file-backed observations '
                '(get_data must cheaply subset pointing fields). Pre-compute the footprint'
                'and pass an explicit WCS landscape instead.'
            )
            raise RuntimeError(msg)
        lc = self.config.landscape
        if lc.wcs is None:
            raise ValueError('WCS landscape config is required for auto footprint scanning.')
        wcs_config = lc.wcs
        res = wcs_config.resolution * pixell.utils.arcmin
        proj = wcs_config.projection.name.lower()
        pointing_fields = [ReaderField.BORESIGHT_QUATERNIONS, ReaderField.DETECTOR_QUATERNIONS]

        n = len(self.observations)
        corners_rad = np.empty((2, 2, n))  # [bottom-left, top-right] corners per observation
        for i, lazy_obs in enumerate(self.observations):
            obs = lazy_obs.get_data(pointing_fields)
            shape, wcs = obs.get_wcs_shape_and_kernel(
                resolution_arcmin=wcs_config.resolution, projection=wcs_config.projection
            )
            # RA _decreases_ left-to-right (increases toward the East)
            # so each corner pair is technically [[dec_lo, ra_hi], [dec_hi, ra_lo]]
            corners_rad[..., i] = pixell.enmap.corners(shape, wcs, corner=True)

        # find the corners of the smallest box covering all the individual patches
        dec_lo = corners_rad[0, 0].min()
        dec_hi = corners_rad[1, 0].max()
        # minimum_enclosing_arc expects [lo, hi] intervals with shape (2, n)
        ra_lo, ra_hi = minimum_enclosing_arc(corners_rad[::-1, 1].T)
        union_box = np.array([[dec_lo, ra_hi], [dec_hi, ra_lo]])

        # create the final shape and WCS objects for this covering box
        shape, wcs = pixell.enmap.geometry(pos=union_box, res=res, proj=proj)
        return WCSLandscape.from_wcs(shape, wcs, lc.stokes, self.config.dtype)


def _static_landscape(lc: LandscapeConfig, dtype: DTypeLike) -> StokesLandscape | None:
    """Build a landscape from config alone, without observation data.

    Returns None if the landscape cannot be determined without scanning observations
    (i.e. WCS with no explicit geometry).
    """
    if lc.healpix is not None:
        return HealpixLandscape(
            nside=lc.healpix.nside,
            stokes=lc.stokes,
            dtype=dtype,
            nested=lc.healpix.ordering == 'nest',
        )
    if lc.wcs is not None and lc.wcs.has_geometry:
        return _wcs_landscape_from_geometry(lc.wcs, lc.stokes, dtype)
    return None


def _wcs_landscape_from_geometry(
    wcs_config: WCSConfig,
    stokes: ValidStokesLiteral,
    dtype: DTypeLike,
) -> WCSLandscape:
    """Build a WCSLandscape from an explicit geometry specification."""
    if not wcs_config.has_geometry:
        raise ValueError('wcs_config must specify a geometry_file or patch.')
    if wcs_config.geometry_file is not None:
        shape, wcs = pixell.enmap.read_map_geometry(wcs_config.geometry_file)
    else:
        assert wcs_config.patch is not None  # ty assert: has_geometry guarantees patch is set here
        res = wcs_config.resolution * pixell.utils.arcmin
        half_w = np.radians(wcs_config.patch.width / 2)
        half_h = np.radians(wcs_config.patch.height / 2)
        ra, dec = np.radians(wcs_config.patch.center)
        corners = np.array([[dec - half_h, ra + half_w], [dec + half_h, ra - half_w]])
        shape, wcs = pixell.enmap.geometry(
            pos=corners, res=res, proj=wcs_config.projection.name.lower()
        )
    return WCSLandscape.from_wcs(shape, wcs, stokes, dtype)


@dataclass
class MapMaker:
    """Class for generic mapmakers which consume GroundObservationData."""

    config: MapMakingConfig
    logger: Logger = furax_logger

    supports_bilinear_pointing: ClassVar[bool] = False
    """Whether this mapmaker solves the map system iteratively (full pixel coupling).

    Only such solvers may use bilinear pointing; the direct block-diagonal binners drop the
    off-diagonal pixel coupling that interpolation introduces and would return a biased map.
    """

    def __post_init__(self) -> None:
        if self.config.pointing.interpolation == 'bilinear' and not self.supports_bilinear_pointing:
            raise ValueError(
                f'{type(self).__name__} does not support bilinear pointing: the direct binned '
                'solver ignores the off-diagonal pixel coupling that interpolation introduces and '
                'would return a biased map. Use the ML mapmaker (method=ML) instead.'
            )
        if self.config.use_templates:
            raise NotImplementedError(f'{type(self).__name__} does not support templates.')

    @abstractmethod
    def make_map(self, observation: AbstractGroundObservation[Any]) -> dict[str, Any]: ...

    def run(
        self, observation: AbstractGroundObservation[Any], out_dir: str | Path | None
    ) -> dict[str, Any]:
        results = self.make_map(observation)

        # Save output
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self._save(results, out_dir)
            self.config.dump_yaml(out_dir / 'mapmaking_config.yaml')
            self.logger.info('Mapmaking config saved to file')

        return results

    def _save(self, results: dict[str, Any], out_dir: Path) -> None:
        for key, m in results.items():
            if isinstance(m, (jax.Array, np.ndarray)):
                np.save(out_dir / key, np.array(m))
            elif isinstance(m, StokesIQU):
                np.save(out_dir / key, np.stack([m.i, m.q, m.u], axis=0))
            elif isinstance(m, pixell.enmap.ndmap):
                pixell.enmap.write_map((out_dir / f'{key}.hdf').as_posix(), m, allow_modify=True)
            elif isinstance(m, WCS):
                header = m.to_header()
                hdu = fits.PrimaryHDU(header=header)
                hdu.writeto(out_dir / f'{key}.fits', overwrite=True)
            elif isinstance(m, StokesLandscape):
                with open(out_dir / f'{key}.pkl', 'wb') as f:
                    pickle.dump(m, f)
            elif isinstance(m, dict):
                self._save(m, out_dir)
                continue
            else:
                continue
            self.logger.info(f'Mapmaking result [{key}] saved to file')

    @classmethod
    def from_config(cls, config: MapMakingConfig, logger: Logger | None = None) -> 'MapMaker':
        """Return the appropriate mapmaker based on the config's mapmaking method."""
        maker = {
            Methods.BINNED: BinnedMapMaker,
            Methods.MAXL: MLMapmaker,
            Methods.POMME: PommeMapMaker,
        }[config.method]

        if logger is None:
            return maker(config)
        else:
            return maker(config, logger=logger)

    @classmethod
    def from_yaml(cls, path: str | Path, logger: Logger | None = None) -> 'MapMaker':
        return cls.from_config(MapMakingConfig.load_yaml(path), logger=logger)

    def get_landscape(self, observation: AbstractGroundObservation[Any]) -> StokesLandscape:
        """Landscape used for mapmaking with given observation."""
        lc = self.config.landscape
        if (landscape := _static_landscape(lc, self.config.dtype)) is not None:
            return landscape
        assert lc.wcs is not None  # ty assert: _static_landscape returns None only for auto WCS
        wcs_shape, wcs_kernel = observation.get_wcs_shape_and_kernel(
            resolution_arcmin=lc.wcs.resolution, projection=lc.wcs.projection
        )
        return WCSLandscape.from_wcs(wcs_shape, wcs_kernel, lc.stokes, self.config.dtype)

    def get_pointing(
        self, observation: AbstractGroundObservation[Any], landscape: StokesLandscape
    ) -> AbstractLinearOperator:
        """Operator containing pointing information for given observation."""
        det_off_ang = observation.get_detector_offset_angles().astype(landscape.dtype)

        if self.config.pointing.on_the_fly:
            pointing = PointingOperator.create(
                landscape,
                Quaternion.from_array(observation.get_boresight_quaternions()),
                Quaternion.from_array(observation.get_detector_quaternions()),
                batch_samples=self.config.pointing.batch_samples,
                interpolate=self.config.pointing.interpolation == 'bilinear',
            )
            return pointing

        else:
            pixel_inds, spin_ang = observation.get_pointing_and_spin_angles(landscape)
            point_ang = spin_ang + det_off_ang[:, None]

            # Index the (trailing) sky axes, leaving the leading Stokes axis of the map intact.
            if isinstance(landscape, WCSLandscape | AstropyWCSLandscape):
                assert pixel_inds.shape[-1] == 2, 'Wrong WCS landscape format'
                indexer = IndexOperator(
                    (..., pixel_inds[..., 0], pixel_inds[..., 1]), in_structure=landscape.structure
                )
            elif isinstance(landscape, HealpixLandscape):
                if pixel_inds.shape[-1] == 1:
                    pixel_inds = pixel_inds[..., 0]
                indexer = IndexOperator((..., pixel_inds), in_structure=landscape.structure)
            else:
                raise NotImplementedError(
                    f'Cannot index a landscape of type {type(landscape).__name__}.'
                )

            # Rotation due to coordinate transform
            tod_shape = pixel_inds.shape[:2]
            rotator = QURotationOperator.create(
                tod_shape, dtype=landscape.dtype, stokes=landscape.stokes, angles=point_ang
            )

            return (rotator @ indexer).reduce()

    def get_acquisition(
        self,
        observation: AbstractGroundObservation[Any],
        landscape: StokesLandscape,
    ) -> AbstractLinearOperator:
        """Acquisition operator mapping sky maps to time-ordered data."""
        pointing = self.get_pointing(observation, landscape)

        if self.config.demodulated:
            return pointing
        else:
            shape = (observation.n_detectors, observation.n_samples)
            stokes = landscape.stokes
            dtype = self.config.dtype
            polarizer = LinearPolarizerOperator.create(
                shape,
                stokes=stokes,
                dtype=dtype,
                angles=jnp.asarray(
                    observation.get_detector_offset_angles().astype(self.config.dtype)[:, None]
                ),
            )
            hwp = HWPOperator.create(
                shape,
                stokes=stokes,
                dtype=dtype,
                angles=jnp.asarray(observation.get_hwp_angles().astype(self.config.dtype)),
            )

            return (polarizer @ hwp @ pointing).reduce()

    def get_scanning_masker(
        self, observation: AbstractGroundObservation[Any]
    ) -> AbstractLinearOperator:
        """Select only the scanning intervals of the given TOD.

        The TOD has shape (ndets, nsamps).
        """
        in_structure = ShapeDtypeStruct(
            shape=(observation.n_detectors, observation.n_samples), dtype=self.config.dtype
        )
        if not self.config.scanning_mask:
            return IdentityOperator(in_structure=in_structure)

        mask = observation.get_scanning_mask()
        out_structure = ShapeDtypeStruct(
            shape=(observation.n_detectors, np.sum(mask)), dtype=self.config.dtype
        )
        masker = IndexOperator(
            (slice(None), jnp.array(mask)),
            in_structure=in_structure,
            out_structure=out_structure,
        )
        return masker

    def get_scanning_mask_projector(
        self, observation: AbstractGroundObservation[Any]
    ) -> AbstractLinearOperator:
        """Zero the values outside the scanning intervals of the given TOD.

        The TOD has shape (ndets, nsamps).
        """
        structure = ShapeDtypeStruct(
            shape=(observation.n_detectors, observation.n_samples), dtype=self.config.dtype
        )
        if not self.config.scanning_mask:
            return IdentityOperator(in_structure=structure)

        # mask is broadcasted along detector axis
        mask = observation.get_scanning_mask()
        return MaskOperator.from_boolean_mask(mask, in_structure=structure)

    def get_sample_mask_projector(
        self, observation: AbstractGroundObservation[Any]
    ) -> AbstractLinearOperator:
        """Zero the given TOD at masked (flagged) samples.

        The TOD has shape (ndets, nsamps).
        """
        structure = ShapeDtypeStruct(
            shape=(observation.n_detectors, observation.n_samples), dtype=self.config.dtype
        )
        if not self.config.sample_mask:
            return IdentityOperator(in_structure=structure)

        # Note the mask value is 1 at valid (unmasked) samples
        mask = observation.get_sample_mask()
        return MaskOperator.from_boolean_mask(mask, in_structure=structure)

    def get_mask_projector(
        self, observation: AbstractGroundObservation[Any]
    ) -> AbstractLinearOperator:
        """Mask operator which incorporates both the scanning and sample mask projectors."""
        return (
            self.get_scanning_mask_projector(observation)
            @ self.get_sample_mask_projector(observation)
        ).reduce()

    def get_or_fit_noise_model(self, observation: AbstractGroundObservation[Any]) -> NoiseModel:
        """Return a noise model for the observation.

        The model type (diagonal, toeplitz, ...) is specified by the mapmaker.
        Attempts to load the noise model from the data if available,
        but otherwise fits a model to the data.
        """
        config = self.config

        if config.weighting.mode == WeightingMode.IDENTITY:
            return WhiteNoiseModel(sigma=jnp.ones(observation.n_detectors, dtype=config.dtype))

        Model = WhiteNoiseModel if config.binned else AtmosphericNoiseModel

        if config.weighting.source == NoiseSource.PRECOMPUTED:
            # Load the noise model from data if available
            noise_model = observation.get_noise_model()
            if noise_model:
                self.logger.info('Loading noise model from data')
                if isinstance(noise_model, Model):
                    return noise_model
                if config.binned and isinstance(noise_model, AtmosphericNoiseModel):
                    return noise_model.to_white_noise_model()
            self.logger.info('No noise model found for loading')

        # Otherwise, fit the noise model from data
        self.logger.info('Fitting noise model from data')
        f, Pxx = jax.scipy.signal.welch(
            jnp.asarray(observation.get_tods()).astype(config.dtype),
            fs=observation.sample_rate,
            nperseg=config.weighting.fitting.nperseg,
        )
        hwp_frequency = jnp.asarray(observation.get_hwp_frequency())
        return Model.fit_psd_model(
            f,
            Pxx,
            sample_rate=jnp.array(observation.sample_rate),
            hwp_frequency=hwp_frequency,
            config=config.weighting.fitting,
        )

    def get_pixel_selector(
        self, blocks: Float[Array, '... nstokes nstokes'], landscape: StokesLandscape
    ) -> IndexOperator:
        """Select indices of map pixels satisfying the hit and condition-number cuts.

        Pixels must meet the minimum fractional hits (hits_cut) and condition number
        (cond_cut) criteria.
        """
        config = self.config

        # eigs = jnp.linalg.eigvalsh(blocks)
        eigs = np.linalg.eigvalsh(blocks)
        hits_quantile = np.quantile(eigs[(eigs[..., -1] > 0),], q=0.95)
        valid = jnp.logical_and(
            eigs[..., -1] > config.hits_cut * hits_quantile,
            eigs[..., 0] > config.cond_cut * eigs[..., -1],
        )
        return IndexOperator((..., *jnp.where(valid)), in_structure=landscape.structure)


class BinnedMapMaker(MapMaker):
    """Class for mapmaking with diagonal noise covariance."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validation on config
        if not self.config.binned:
            raise ValueError('Binned Mapmaker is incompatible with binned=False')

    def make_map(self, observation: AbstractGroundObservation[Any]) -> dict[str, Any]:
        config = self.config
        logger_info = lambda msg: self.logger.info(f'Binned Mapmaker: {msg}')

        # Data and landscape
        data = jnp.asarray(observation.get_tods(), dtype=config.dtype)
        data_struct = ShapeDtypeStruct(data.shape, data.dtype)
        landscape = self.get_landscape(observation)

        # Acquisition (I, Q, U Maps -> TOD)
        acquisition = self.get_acquisition(observation, landscape=landscape)
        logger_info('Created acquisition operator')

        # Optional mask for scanning
        masker = self.get_scanning_masker(observation)
        acquisition = masker @ acquisition
        data_struct = masker.out_structure  # Now with a subset of samples
        logger_info('Created scanning mask operator')

        # Noise
        noise_model = self.get_or_fit_noise_model(observation)
        inv_noise = noise_model.inverse_operator(data_struct)
        logger_info('Created inverse noise covariance operator')

        # System matrix
        system = BJPreconditioner.create((acquisition.T @ inv_noise @ acquisition).reduce())
        logger_info('Created system operator')

        # Mapmaking operator
        binner = acquisition.T @ inv_noise @ masker
        mapmaking_operator = system.inverse() @ binner

        @jax.jit
        def process(d):
            return mapmaking_operator.reduce()(d)

        logger_info('Set up mapmaking operator')

        # Run mapmaking
        res = process(data)
        res.i.block_until_ready()
        logger_info('Finished mapmaking')

        if config.debug:
            res = process(data)
            res.i.block_until_ready()
            logger_info('Test - second time - Finished mapmaking')

        final_map = np.array([res.i, res.q, res.u])
        weights = np.array(system.blocks)

        output = {'map': final_map, 'weights': weights}
        if isinstance(landscape, WCSLandscape):
            output['wcs'] = landscape.to_wcs()
        elif isinstance(landscape, AstropyWCSLandscape):
            output['wcs'] = landscape.wcs
        if (
            config.weighting.source == NoiseSource.FIT
            and config.weighting.mode != WeightingMode.IDENTITY
        ):
            output['noise_fit'] = noise_model.to_array()
        if config.debug:
            proj_map = (masker.T @ acquisition)(res)
            output['proj_map'] = proj_map

        return output


class MLMapmaker(MapMaker):
    """Class for mapmaking with maximum likelihood (ML) estimator."""

    supports_bilinear_pointing: ClassVar[bool] = True
    """The ML solve is iterative, so it accounts for the pixel coupling from interpolation."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validation on config
        if self.config.demodulated:
            raise ValueError('ML Mapmaker is incompatible with demodulated=True')

    def make_map(self, observation: AbstractGroundObservation[Any]) -> dict[str, Any]:
        config = self.config
        logger_info = lambda msg: self.logger.info(f'ML Mapmaker: {msg}')

        # Data and landscape
        data = jnp.asarray(observation.get_tods(), dtype=config.dtype)
        data_struct = ShapeDtypeStruct(data.shape, data.dtype)
        landscape = self.get_landscape(observation)

        # Acquisition (I, Q, U Maps -> TOD)
        acquisition = self.get_acquisition(observation, landscape=landscape)
        logger_info('Created acquisition operator')

        # Optional mask for scanning
        masker = self.get_mask_projector(observation)
        valid_sample_fraction = (
            1.0
            if isinstance(masker, IdentityOperator)
            else float(jnp.mean(masker(jnp.ones(data.shape, data.dtype))))
        )
        logger_info('Created mask operator')
        logger_info(f'Valid sample fraction: {valid_sample_fraction:.4f}')

        # Noise
        noise_model = self.get_or_fit_noise_model(observation)
        inv_noise = noise_model.inverse_operator(
            data_struct,
            sample_rate=observation.sample_rate,
            correlation_length=config.weighting.correlation_length,
        )
        noise = noise_model.operator(
            data_struct,
            sample_rate=observation.sample_rate,
            correlation_length=config.weighting.correlation_length,
        )
        logger_info('Created noise and inverse noise covariance operators')

        # Approximate system matrix with diagonal noise covariance and full map pixels
        diag_inv_noise: AbstractLinearOperator
        if isinstance(inv_noise, SymmetricBandToeplitzOperator):
            # Correlated (Toeplitz) weighting: use the zero-lag (diagonal) band value.
            diag_inv_noise = DiagonalOperator(
                inv_noise.band_values[..., [0]], in_structure=data_struct
            )
        elif inv_noise.is_diagonal:
            # Identity / diagonal (white) weighting.
            diag_inv_noise = inv_noise
        else:
            raise NotImplementedError(
                f'Cannot approximate {type(inv_noise).__name__} by a diagonal operator.'
            )
        diag_system = BJPreconditioner.create(acquisition.T @ diag_inv_noise @ masker @ acquisition)
        logger_info('Created approximate system matrix')

        # Map pixel selection
        blocks = diag_system.blocks
        selector = self.get_pixel_selector(blocks, landscape)
        logger_info(
            f'Selected {prod(selector.out_structure.shape)}\
                            /{prod(landscape.shape)} pixels'
        )

        # Adjust the sample mask according to the new pixel selection
        positive_sample_hits = (
            (masker @ acquisition @ selector.T)(
                StokesIQU.from_stokes(
                    i=jnp.ones(selector.out_structure.shape, dtype=data.dtype),
                    q=jnp.zeros(selector.out_structure.shape, dtype=data.dtype),
                    u=jnp.zeros(selector.out_structure.shape, dtype=data.dtype),
                )
            )
            > 0
        ).astype(data.dtype)
        masker = DiagonalOperator(positive_sample_hits, in_structure=data_struct)
        logger_info(f'Updated valid sample fraction: {jnp.mean(masker._diagonal):.4f}')

        # Preconditioner
        # We use the approximate diagonal system matrix before the mask update
        preconditioner = selector @ diag_system.inverse() @ selector.T
        h = acquisition @ selector.T

        if config.gaps.treatment != GapTreatment.NESTED:
            M = masker @ inv_noise @ masker
        else:
            nested_solver = lineax.CG(
                rtol=config.gaps.nested.rtol,
                atol=config.gaps.nested.atol,
                max_steps=config.gaps.nested.inner_steps,
            )
            M = (
                masker
                @ (masker @ noise @ masker).I(
                    solver=nested_solver,
                    preconditioner=masker @ inv_noise @ masker,
                )
                @ masker
            )
            logger_info('Set up nested PCG for the noise inverse')

        solver = lineax.CG(**config.solver.options)
        options = {'solver': solver, 'preconditioner': preconditioner}
        mapmaking_operator = (h.T @ M @ h).I(**options) @ h.T @ M

        @jax.jit
        def process(d):
            return mapmaking_operator.reduce()(d)

        logger_info('Completed setting up the solver')

        # Run mapmaking
        rec_map = process(data)
        result_map = selector.T(rec_map)
        result_map.i.block_until_ready()
        logger_info('Finished mapmaking computation')

        # Get weights after pixel selection
        weights = jnp.zeros_like(blocks)
        weights = weights.at[selector.indices + (slice(None), slice(None))].add(
            blocks[selector.indices + (slice(None), slice(None))]
        )

        # Format output and compute auxiliary data
        final_map = np.array([result_map.i, result_map.q, result_map.u])

        output = {'map': final_map, 'weights': weights, 'weights_uncut': blocks}
        if isinstance(landscape, WCSLandscape):
            output['wcs'] = landscape.to_wcs()
        elif isinstance(landscape, AstropyWCSLandscape):
            output['wcs'] = landscape.wcs
        if (
            config.weighting.source == NoiseSource.FIT
            and config.weighting.mode != WeightingMode.IDENTITY
        ):
            output['noise_fit'] = noise_model.to_array()
        if config.debug:
            proj_map = (masker @ acquisition)(result_map)
            output['projs'] = {'proj_map': proj_map}

        return output


class PommeMapMaker(MapMaker):
    """Class for Pomme mapmaking with diagonal noise covariance."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validation on config
        if not self.config.binned:
            raise ValueError('Pomme Mapmaker is currently incompatible with binned=False')
        if self.config.pomme_tau < 2:
            raise ValueError('Pomme tau should be at least 2')
        if self.config.landscape.stokes != 'QU':
            raise ValueError('Pomme only compatible with stokes=QU')

    def make_map(self, observation: AbstractGroundObservation[Any]) -> dict[str, Any]:
        config = self.config
        logger_info = lambda msg: self.logger.info(f'Pomme Mapmaker: {msg}')

        # Data and landscape
        data = jnp.asarray(observation.get_tods(), dtype=config.dtype)
        data_struct = ShapeDtypeStruct(data.shape, data.dtype)
        landscape = self.get_landscape(observation)

        # Acquisition (I, Q, U Maps -> TOD)
        acquisition = self.get_acquisition(observation, landscape=landscape)
        logger_info('Created acquisition operator')

        # Pomme projector
        pomme_projector = PommeProjectionOperator(self.config.pomme_tau, in_structure=data_struct)

        # Optional mask for scanning
        masker = self.get_mask_projector(observation)
        valid_sample_fraction = (
            1.0
            if isinstance(masker, IdentityOperator)
            else float(jnp.mean(masker(jnp.ones(data.shape, data.dtype))))
        )
        logger_info('Created mask operator')
        logger_info(f'Valid sample fraction: {valid_sample_fraction:.4f}')

        # Additionally, mask all tau-intervals with any masked samples
        tau_mask = jnp.abs(pomme_projector(masker(jnp.ones_like(data)))) < 0.5 / config.pomme_tau
        masker @= MaskOperator.from_boolean_mask(tau_mask, in_structure=data_struct)
        valid_sample_fraction = float(jnp.mean(masker(jnp.ones(data.shape, data.dtype))))
        logger_info(f'Updated valid sample fraction: {valid_sample_fraction:.4f}')

        # Noise
        noise_model = self.get_or_fit_noise_model(observation)
        inv_noise = noise_model.inverse_operator(data_struct)
        logger_info('Created inverse noise covariance operator')

        # Approximate system matrix with diagonal noise covariance and full map pixels
        diag_system = BJPreconditioner.create(
            (acquisition.T @ inv_noise @ masker @ acquisition).reduce()
        )
        logger_info('Created approximate system matrix')

        # Map pixel selection
        blocks = diag_system.blocks
        selector = self.get_pixel_selector(blocks, landscape)
        logger_info(
            f'Selected {prod(selector.out_structure.shape)}\
                            /{prod(landscape.shape)} pixels'
        )

        # Preconditioner
        preconditioner = selector @ diag_system.inverse() @ selector.T

        # Mapmaking operators
        h = acquisition @ selector.T
        mp = masker
        ap = inv_noise @ pomme_projector
        lhs = h.T @ mp @ ap @ mp @ h
        rhs_op = jax.jit(lambda d: (h.T @ mp @ ap @ mp).reduce()(d))

        solver = lineax.CG(**self.config.solver.options)
        spd = OperatorTag.POSITIVE_SEMIDEFINITE
        lx_system = as_lineax_operator(lhs, spd)
        lx_precond = as_lineax_operator(preconditioner.reduce(), spd)
        logger_info('Completed setting up the solver')

        # Run mapmaking
        rhs = rhs_op(data)
        y0 = preconditioner(rhs)
        solution = lineax.linear_solve(
            lx_system,
            rhs,
            solver=solver,
            options={'preconditioner': lx_precond, 'y0': y0},
            throw=False,
        )
        result_map = selector.T(solution.value)
        result_map.q.block_until_ready()
        num_steps = solution.stats['num_steps']
        logger_info(f'Finished mapmaking computation. Number of PCG steps: {num_steps}')

        # Format output and compute auxiliary data
        final_map = np.array([result_map.q, result_map.u])

        output = {'map': final_map, 'weights': blocks}
        if isinstance(landscape, AstropyWCSLandscape):
            output['wcs'] = landscape.wcs
        if (
            config.weighting.source == NoiseSource.FIT
            and config.weighting.mode != WeightingMode.IDENTITY
        ):
            output['noise_fit'] = noise_model.to_array()
        if config.debug:
            proj_map = (mp @ acquisition)(result_map)
            output['proj_map'] = proj_map

        return output


class _IQUModulationOperator(AbstractLinearOperator):
    """Add the input Stokes signals into a single HWP-modulated signal.

    Similar to ``LinearPolarizerOperator @ QURotationOperator(hwp_angle)``, except that
    only half of the QU rotation needs to be computed.
    """

    cos_hwp_angle: Float[Array, ' samps']
    sin_hwp_angle: Float[Array, ' samps']

    def __init__(
        self,
        shape: tuple[int, ...],
        hwp_angle: Float[Array, '...'],
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        in_structure = Stokes.class_for('IQU').structure_for(shape, dtype)
        object.__setattr__(self, 'cos_hwp_angle', jnp.cos(4 * hwp_angle.astype(dtype)))
        object.__setattr__(self, 'sin_hwp_angle', jnp.sin(4 * hwp_angle.astype(dtype)))
        object.__setattr__(self, 'in_structure', in_structure)

    def mv(self, x: StokesType) -> Float[Array, '...']:
        return x.i + self.cos_hwp_angle[None, :] * x.q + self.sin_hwp_angle[None, :] * x.u


class _QUModulationOperator(AbstractLinearOperator):
    """Add the input Stokes signals into a single HWP-modulated signal.

    Similar to ``LinearPolarizerOperator @ QURotationOperator(hwp_angle)``, except that
    only half of the QU rotation needs to be computed.
    """

    cos_hwp_angle: Float[Array, ' samps']
    sin_hwp_angle: Float[Array, ' samps']

    def __init__(
        self,
        shape: tuple[int, ...],
        hwp_angle: Float[Array, '...'],
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        in_structure = Stokes.class_for('QU').structure_for(shape, dtype)
        object.__setattr__(self, 'cos_hwp_angle', jnp.cos(4 * hwp_angle.astype(dtype)))
        object.__setattr__(self, 'sin_hwp_angle', jnp.sin(4 * hwp_angle.astype(dtype)))
        object.__setattr__(self, 'in_structure', in_structure)

    def mv(self, x: StokesType) -> Float[Array, '...']:
        return self.cos_hwp_angle[None, :] * x.q + self.sin_hwp_angle[None, :] * x.u


def _parameter_name(path: tuple[Any, ...]) -> str:
    """The name of a leaf of a noise model, from its path: `sigma`, or `<key>/sigma` if nested."""
    return '/'.join(str(getattr(k, 'name', getattr(k, 'key', k))) for k in path)


def _solver_stats(result: furax.linalg.CGResult, options: dict[str, Any]) -> dict[str, Any]:
    """The statistics of a CG solve: its convergence, residuals and options, as JSON values.

    A non-finite residual, from a diverged solve, is `None`: JSON has no NaN or infinity.
    """
    n_steps = int(result.num_steps)
    # residuals[0] is the initial residual, residuals[i + 1] the one after step i
    residuals = np.asarray(result.residuals)[: n_steps + 1]
    return {
        'num_steps': n_steps,
        'converged': bool(result.converged),
        'residuals': [float(r) if np.isfinite(r) else None for r in residuals],
        'options': options,
    }
