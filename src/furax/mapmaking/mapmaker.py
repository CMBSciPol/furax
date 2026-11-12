import logging
import math
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import replace
from functools import cached_property, partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from fastquat import Quaternion
from jax.experimental import multihost_utils as mhu
from jaxtyping import Array, Bool, Float, Int64, Integer, PyTree

import furax.linalg
import furax.tree
from furax.core import HostEnsemble, IndexOperator, Policy, StackedEnsemble, map_reduce
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, StokesI, StokesType
from furax.profiling import format_bytes

from ._accumulated import AccumulatedModel
from ._bucket import BucketModel, Tile, bucket_apply
from ._catalog import ObservationCatalog, Read, detector_table, gather_rows
from ._diagnostics import modes_correction, tile_statistics
from ._gaps import GapSplines, gap_budget, knot_spacing
from ._injection import inject, load_sky, unpadded
from ._landscape import build_landscape
from ._model import (
    FACTORED_KINDS,
    ObservationModel,
    ObservationTemplates,
    acquisition_operator,
    restrict_streams,
)
from ._observation import (
    AbstractLazyObservation,
    ObservationBufferShape,
    ObservationProbe,
    ReaderField,
)
from ._provenance import provenance
from ._reader import ObservationReader, TileReader, detector_tile_structure
from .config import (
    DiagnosticsConfig,
    GapTreatment,
    InjectionMode,
    MapMakingConfig,
    NoiseSource,
    WeightingMode,
)
from .gap_filling import gap_fill
from .layout import TileBucket, TileLayout, TilePlacement
from .lowrank import LowRankNoiseModel
from .results import CONFIG_FILE, PROVENANCE_FILE, MapMakingResults, write_json
from .templates.gram import factor
from .weight import WeightOperator

logger = logging.getLogger(__name__)

# Relative spread of the observation sample rates the low-rank weighting accepts as one rate.
_SAMPLE_RATE_TOLERANCE = 1e-3


class MultiObservationMapMaker[T]:
    """Class for mapping multiple observations together."""

    def __init__(
        self,
        observations: Sequence[AbstractLazyObservation[T]],
        config: MapMakingConfig | None = None,
    ) -> None:
        self.observations = observations
        self.config = config or MapMakingConfig()  # use defaults if not provided
        self.landscape = build_landscape(
            self.observations, self.config.landscape, self.config.dtype
        )
        # Build the readers once, up front: they only need the probe shapes (no mesh context)
        rhs_fields = {ReaderField.METADATA, ReaderField.SAMPLE_DATA}
        model_fields = ObservationModel.required_reader_fields(self.config)
        template_fields = ObservationTemplates.required_reader_fields(self.config)
        self.reader_fields = model_fields | rhs_fields | template_fields
        self.readers = self.get_readers(self.reader_fields)

    @cached_property
    def sky(self) -> StokesType | None:
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
        gaps = self.config.gaps
        # the gap amplitudes are counted only to size them (`ExplicitGapsConfig.probe`)
        count_gaps = self._explicit_gaps and gaps.explicit.probe
        resolution = gaps.explicit.resolution if count_gaps else None
        fraction = gaps.max_flagged_fraction or 1.0
        local: dict[int, ObservationProbe | str] = {}
        for i in range(jax.process_index(), self.n_observations, n_proc):
            try:
                local[i] = self.observations[i].probe(need_intervals, resolution, fraction)
            except Exception as error:
                logger.exception('Probe of observation %d failed', i)
                local[i] = f'{type(error).__name__}: {error}'
        names = [observation.name for observation in self.observations]
        return ObservationCatalog(names, gather_rows(local))

    @cached_property
    def _probe_shapes(self) -> list[ObservationBufferShape]:
        """Every observation's buffer shape.

        Observations whose probe failed are skipped by the readers and gated out of the
        accumulation. They inherit the largest successful probe shape so they cannot form an
        undersized bucket of their own.

        The sample count is raised to `_minimum_buffer_samples`, so the buckets and the readers
        size their buffers identically.
        """
        rows = self._catalog.probed
        failed = self._catalog.probe_failed
        rows[failed] = rows[~failed].max(axis=0) if np.any(~failed) else (1, 1, 0)
        minimum_samples = self._minimum_buffer_samples
        return [
            ObservationBufferShape(int(row[0]), max(int(row[1]), minimum_samples), int(row[2]))
            for row in rows
        ]

    @cached_property
    def _bridge_samples(self) -> int:
        """The padding every observation gets after its data, for the circulant weight to wrap.

        See [`CirculantBridgeConfig`][furax.mapmaking.config.CirculantBridgeConfig].
        """
        weighting = self.config.weighting
        if (
            weighting.mode != WeightingMode.LOW_RANK
            or (bridge := weighting.low_rank.bridge) is None
        ):
            return 0
        return max(math.ceil(bridge.length * self._probed_sample_rate), bridge.amplitudes)

    @cached_property
    def _buffer_shapes(self) -> list[ObservationBufferShape]:
        """The shapes the buckets and the readers size their buffers from.

        These are the probed shapes, with the padding the circulant weight needs after the data.
        """
        shapes = [
            shape._replace(sample_count=shape.sample_count + self._bridge_samples)
            for shape in self._probe_shapes
        ]
        if not self.config.fft_friendly_sample_count:
            return shapes
        return [shape.fft_friendly() for shape in shapes]

    @property
    def _minimum_buffer_samples(self) -> int:
        """Smallest sample envelope accepted by the configured TOD operators."""
        minimum = self.config.pomme_tau if self.config.pomme_tau is not None else 1
        if self.config.weighting.mode == WeightingMode.TOEPLITZ:
            correlation_length = self.config.weighting.correlation_length
            if self.config.gaps.treatment == GapTreatment.FILL:
                # The constrained realization embeds the symmetric band in a circulant kernel.
                correlation_length = 2 * correlation_length - 1
            minimum = max(minimum, correlation_length)
        return minimum

    @cached_property
    def layout(self) -> TileLayout:
        """How the observations are bucketed, split into tiles and laid out over the devices."""
        return TileLayout.create(
            self._buffer_shapes,
            n_devices=jax.device_count(),
            max_buckets=self.config.max_buckets,
            detector_tile_size=self.config.detector_tile_size,
        )

    @property
    def _explicit_gaps(self) -> bool:
        """Whether the gaps are explicit, their amplitudes solved for with the map."""
        return self.config.gaps.treatment == GapTreatment.EXPLICIT and not self.config.binned

    @cached_property
    def _gap_splines(self) -> list[GapSplines | None]:
        """The spline model of the explicit gaps of each bucket; `None` without.

        The knot spacing of a bucket, in samples, is `ExplicitGapsConfig.resolution` at the sample
        rates its observations probed, the finest if they differ. Its budget of amplitudes fits
        every observation in it: from the counts its observations probed, when they all did at
        that spacing, or from the bound that holds for any flags (see
        [`gap_budget`][furax.mapmaking._gaps.gap_budget]).
        """
        if not self._explicit_gaps:
            return [None] * len(self.layout.buckets)
        gaps = self.config.gaps
        assert gaps.max_flagged_fraction is not None  # required by GapsConfig
        probes = self._catalog.probes
        shapes = self._probe_shapes
        models: list[GapSplines | None] = []
        for b, bucket in enumerate(self.layout.buckets):
            # an observation whose probe failed is not read
            read = {
                i: p for i in bucket.observations if isinstance(p := probes[i], ObservationProbe)
            }
            spacings = {i: self._knot_spacing(i, p) for i, p in read.items()}
            spacing = min(spacings.values(), default=1)
            counts = {i: p.gap_amplitudes for i, p in read.items()}
            if read and None not in counts.values() and set(spacings.values()) == {spacing}:
                budget = max(count for count in counts.values() if count is not None)
            else:
                budget = max(
                    gap_budget(shapes[i].sample_count, gaps.max_flagged_fraction)
                    for i in bucket.observations
                )
            budget = max(budget, 1)  # the basis indexes its slots: an empty budget has none
            logger.info(
                f'Bucket {b}: {budget} gap amplitudes per detector, knots every {spacing} samples'
            )
            bridge = 0
            if self._bridge_samples:
                assert self.config.weighting.low_rank.bridge is not None
                bridge = self.config.weighting.low_rank.bridge.amplitudes
            models.append(GapSplines(spacing, budget, bridge))
        return models

    def _knot_spacing(self, i: int, probe: ObservationProbe) -> int:
        """The knot spacing of the explicit gaps of observation `i`, in samples."""
        if probe.sample_rate is None:
            msg = (
                f'Observation {self._catalog.names[i]}: explicit gaps need its sample rate, '
                'which its probe does not report'
            )
            raise ValueError(msg)
        return knot_spacing(self.config.gaps.explicit.resolution, probe.sample_rate)

    @cached_property
    def placement(self) -> TilePlacement:
        """The tiles of the layout, placed on the devices of the job."""
        return TilePlacement.create(self.layout)

    @cached_property
    def fit_config(self) -> MapMakingConfig:
        """The run configuration, with the noise fit options completed from the probes.

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
        if weighting.source != NoiseSource.FIT or weighting.mode in (
            WeightingMode.IDENTITY,
            WeightingMode.LOW_RANK,  # fitted on the whole observation
        ):
            return self.config  # no Welch PSD is estimated, so the segment length is unused
        # not `_buffer_shapes`: a length padded further would admit a segment reaching the padding
        shortest = min(shape.sample_count for shape in self._probe_shapes)
        nperseg = min(weighting.fitting.nperseg, shortest)
        if nperseg == weighting.fitting.nperseg:
            return self.config
        logger.info(
            f'Welch segment clipped to {nperseg} samples '
            f'(configured {weighting.fitting.nperseg}) by the shortest observation'
        )
        fitting = replace(weighting.fitting, nperseg=nperseg)
        return replace(self.config, weighting=replace(weighting, fitting=fitting))

    @cached_property
    def _probed_sample_rate(self) -> float:
        """The sample rate shared by the observations, from their probes.

        The low-rank noise model places its frequency bins at it, which must be known before
        compilation.

        An observation whose probe failed is left out here; its read fails later as well.
        """
        rates = np.array(
            [
                p.sample_rate
                for p in self._catalog.probes.values()
                if isinstance(p, ObservationProbe) and p.sample_rate is not None
            ]
        )
        if rates.size == 0:
            raise RuntimeError('no observation probed its sample rate')
        median = float(np.median(rates))
        if np.ptp(rates) > _SAMPLE_RATE_TOLERANCE * median:
            raise ValueError(
                f'the observations have sample rates from {rates.min()} to {rates.max()} Hz: the '
                'low-rank weighting needs a common one'
            )
        logger.info(f'Low-rank noise bins placed at the probed sample rate of {median:.6g} Hz')
        return median

    def get_readers(self, required_fields: Collection[str]) -> tuple[ObservationReader[T], ...]:
        """Build one reader per bucket, over that bucket's observations only.

        A reader reads its bucket's observations in the order of their indices, into buffers of
        whole tiles sized by exactly the observations it can be asked to read.
        """
        shapes = self._buffer_shapes
        failed = self._catalog.probe_failed
        return tuple(
            ObservationReader.from_observations(
                [self.observations[i] for i in bucket.observations],
                requested_fields=required_fields,
                stokes=self.config.read_stokes if self.config.demodulated else None,
                dtype=self.config.dtype,
                sample_dtype=jnp.float32,
                shapes=[shapes[i].whole_tiles(bucket.tile_size) for i in bucket.observations],
                known_failures=np.flatnonzero(failed[bucket.observations]).tolist(),
            )
            for bucket in self.layout.buckets
        )

    def run(self, out_dir: str | Path, resume: bool = False) -> MapMakingResults:
        """Runs the mapmaker and return results after saving them to the given directory.

        The configuration and the provenance are written to `out_dir` before the maps are
        computed.

        Args:
            out_dir: Directory of the products, and of the checkpoints of the solves.
            resume: Resume the solves from the checkpoints in `out_dir`, if any.
        """
        # Only process 0 writes, and so only process 0 spends the git calls on the provenance
        is_writer = jax.process_index() == 0
        out_dir = Path(out_dir).resolve()
        if is_writer:
            out_dir.mkdir(parents=True, exist_ok=True)
            self.config.dump_yaml(out_dir / CONFIG_FILE)
            write_json(out_dir / PROVENANCE_FILE, provenance())
            logger.info(f'Saved mapmaking configuration and provenance to {out_dir}')

        # Loaded now, once the run is on record, so that a bad injection fails before the passes
        # over the data, and shows in the output directory
        _ = self.sky
        results = self.make_maps(checkpoint_dir=out_dir, resume=resume)

        # Save outputs on process 0 only (all processes hold the same maps)
        if is_writer:
            results.save(out_dir)
            logger.info(f'Saved results to {out_dir}')

        # Barrier so other ranks don't race ahead while rank 0 is still writing.
        mhu.sync_global_devices('mapmaker.run.save_done')

        return results

    def _log_layout(self) -> None:
        """Log the device layout and what the bucketing costs in padding (rank 0 only)."""
        if jax.process_index() != 0:
            return
        layout = self.layout
        logger.info(
            f'Layout procs={jax.process_count()} dev_per_proc={len(self.placement.local_devices)} '
            f'dev_total={layout.n_devices} platform={jax.default_backend()}'
        )
        # Every tile holds its detectors' fields, every segment once what its tiles share.
        real_bytes = sum(reader.total_nbytes for reader in self.readers)
        padded_bytes = 0
        for b, (bucket, reader) in enumerate(zip(layout.buckets, self.readers, strict=True)):
            tile, strides = detector_tile_structure(reader.out_structure, bucket.tile_size)
            tile_bytes = furax.tree.nbytes(tile)
            held = {1: bucket.n_tiles, 'segment': bucket.n_segments}
            bucket_bytes = sum(
                furax.tree.nbytes(leaf) * held[stride]
                for leaf, stride in zip(
                    jax.tree.leaves(tile), jax.tree.leaves(strides), strict=True
                )
            )
            padded_bytes += bucket_bytes
            logger.info(
                f'Bucket {b}: {bucket.summary()} '
                f'tile_size={format_bytes(tile_bytes)} '
                f'real_size={format_bytes(reader.total_nbytes)} '
                f'pad_size={format_bytes(bucket_bytes - reader.total_nbytes)}'
            )
        byte_overhead = (padded_bytes - real_bytes) / real_bytes
        logger.info(
            f'Dataset {layout.summary()} real={format_bytes(real_bytes)} '
            f'padded={format_bytes(padded_bytes)} byte_overhead=+{byte_overhead:.1%}'
        )

    def make_maps(
        self, checkpoint_dir: str | Path | None = None, resume: bool = False
    ) -> MapMakingResults:
        """Computes the mapmaker results (maps and other products).

        Args:
            checkpoint_dir: Directory where the CG solves save their iterate every
                [`SolverConfig.checkpoint_every`][furax.mapmaking.config.SolverConfig.checkpoint_every]
                steps. A solve that converges deletes its checkpoint; one that runs out of steps
                saves its last iterate, to be resumed with more steps.
            resume: Start the solves from the checkpoints in `checkpoint_dir`, if any. A checkpoint
                of a system of other shapes, e.g. from other observations, is ignored.
        """
        self._log_layout()

        with jax.set_mesh(self.placement.mesh):
            # Single read pass, so each observation is read/preprocessed exactly once.
            start = time.perf_counter()
            acc = self.build_model_and_accumulate()
            jax.block_until_ready(acc)
            logger.info(f'Finished the pass over the data in {time.perf_counter() - start:.0f} s')

            self._catalog.reads = self._gather_reads()
            if failures := self._catalog.failures():
                # The catalog is the same on every process: one warning is enough
                level = logging.WARNING if jax.process_index() == 0 else logging.DEBUG
                logger.log(level, _excluded_message(failures))
            drops = acc.detector_drops(self.placement)
            if n_dropped := int(drops['dropped'].sum()):
                logger.info(
                    f'Dropped {n_dropped} detectors flagged above '
                    f'{self.config.gaps.max_flagged_fraction:.0%} '
                    f'in {int(drops["dropped"].any(axis=-1).sum())} observations'
                )
            if n_over := int(drops['over_gap_budget'].sum()):
                # the budget is meant to fit every detector kept: this one did not
                logger.warning(
                    f'Dropped {n_over} detectors needing more gap amplitudes than the budget '
                    f'in {int(drops["over_gap_budget"].any(axis=-1).sum())} observations'
                )
            noise_parameters = None
            if self.config.weighting.mode != WeightingMode.IDENTITY:
                noise_parameters = acc.noise_parameters(self.placement)

            BJ = acc.block_jacobi(self.config.binned, self.landscape.structure)
            icov = BJ.blocks.block_until_ready()
            logger.info('Computed the block-Jacobi preconditioner')

            valid_pixels = self.pixel_selection(acc.hit_map, icov)
            # Select valid pixels on the (trailing) sky axes, leaving the leading Stokes axis intact.
            S = IndexOperator(
                (..., *jnp.where(valid_pixels)), in_structure=self.landscape.structure
            )
            n_observed = jnp.sum(acc.hit_map > 0)
            logger.info(
                f'Selected {jnp.sum(valid_pixels)} pixels ({n_observed} seen, '
                f'{valid_pixels.size} total)'
            )

            response_sky = None if self._scanned_sky is not None else self.sky
            system = acc.build_system(S, BJ, response_sky)
            if (two_level := self.config.solver.two_level) is not None:
                system = system.deflate(two_level)
            solved = system.solve(
                self.config.solver,
                self.placement,
                None if checkpoint_dir is None else Path(checkpoint_dir),
                resume,
            )
            sky, amplitudes, blocks = solved.x, None, []
            if system.has_amplitudes:
                sky, *blocks = solved.x
                amplitudes = acc.template_amplitudes(
                    blocks, self.placement, self.config.gaps.explicit.save_amplitudes
                )

            statistics = residual_map = None
            if (diagnostics := self.config.diagnostics_config) is not None:
                start = time.perf_counter()
                statistics, residual_rhs = self._residual_statistics(
                    acc, S.T(sky), blocks, diagnostics
                )
                if residual_rhs is not None:
                    # binned under the white weight, the system of the block-Jacobi preconditioner
                    residual_map = S.T((S @ BJ.I @ S.T).reduce()(S(residual_rhs)))
                message = f'Residual pass in {time.perf_counter() - start:.0f} s'
                if 'chi2' in statistics:
                    chi2, good = statistics['chi2'].sum(), statistics['good'].sum()
                    message += f': chi2/sample={chi2 / max(good, 1):.4f} over {int(good)} samples'
                logger.info(message)

        # Built only now, so that their copies are not held through the solve
        return MapMakingResults(
            map=S.T(sky),  # all sky pixels including those not estimated (zero)
            icov=jnp.moveaxis(icov, [-2, -1], [0, 1]),  # (*pixels, ns, ns) → (ns, ns, *pixels)
            hit_map=jnp.where(valid_pixels, acc.hit_map, 0),  # excluded pixels have zero hits
            solver_stats=solved.solver_stats,
            landscape=self.landscape,
            template_amplitudes=amplitudes,
            injected_map=None if solved.injected_sky is None else S.T(solved.injected_sky),
            residual_map=residual_map,
            observations=self._catalog.to_table(drops, statistics),
            # the noise model covers the streams of the model's TOD, not all those read
            detectors=detector_table(
                self._catalog.reads,
                noise_parameters,
                acc.buckets[0].tiles.template.model.tod_format.stokes,
                drops,
                None if diagnostics is None else _detector_statistics(statistics, diagnostics),
            ),
        )

    def _gather_reads(self) -> dict[int, Read]:
        """What the last pass read of each observation, or the exception that stopped its load.

        Keyed by observation index, gathered from the readers of every process. An observation
        that was not loaded, such as one whose probe failed, is absent.
        """
        local: dict[int, Read] = {}
        for bucket, reader in zip(self.layout.buckets, self.readers, strict=True):
            # the reader reads the observations of its bucket in the order of their indices
            for k, record in reader.records.items():
                local[int(bucket.observations[k])] = record
            for k, error in reader.errors.items():
                local[int(bucket.observations[k])] = error
        return gather_rows(local)

    def build_model_and_accumulate(self) -> AccumulatedModel:
        """Build the model and accumulate the hit map and map RHS in one pass over the data.

        Padding tiles and failed observations are gated out using a zero-mask, so they drop from
        the system.

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
        return AccumulatedModel(buckets=tuple(bucket_models), hit_map=hit_map, map_rhs=map_rhs)

    def _accumulate_bucket(
        self, bucket_index: int, reader: ObservationReader[T], sky: StokesType | None
    ) -> tuple[Int64[Array, '...'], StokesType, BucketModel]:
        """One bucket's pass over the data: its hit map and RHS partials, and its stacked model.

        The pass is an engine call on a host ensemble of the bucket's tiles: each round launches
        one tile on each device while its process reads the next round. An observation is read
        once, when its first tile is, and held on the host until its last tile has been read, so
        only one observation per device is in memory at a time, and the processes need not keep
        in step until the device sum at the end. The stacked outputs land sharded over the tiles
        and the segments, which is the layout `BucketModel` holds.
        """
        bucket = self.layout.buckets[bucket_index]
        tile_reader, read_tile = self._tile_reader(bucket, reader)
        ensemble = HostEnsemble(
            bucket.n_tiles,
            read_tile,
            tile_reader.structure,
            policy=Policy(axis=self.placement.axis),
            prefetch=self._prefetches_reads,
            strides=tile_reader.strides,
            owner=bucket.owner,
            segments=bucket.n_segments,
        )
        # The sums run over the tiles; of the rest, what the tiles of an observation share
        # (boresight pointing, common template bases, ...) is kept once per segment. The Gram
        # inverses and right-hand sides of the templates go where their amplitudes live. A Gram
        # summed over the tiles of an observation is paired with its coarse amplitudes, which
        # follow from the bases alone, the same in every tile.
        spec = {
            **dict.fromkeys(('hits', 'map'), 'sum'),
            **dict.fromkeys(('model', 'implicit', 'operators'), 'auto'),
            'inverses': 1,
            'grams': {
                kind: ('segment_sum', 'segment')
                for kind, held in FACTORED_KINDS.items()
                if held == 'sum'
            },
            'rhs': {
                kind: 'segment_sum' if held == 'sum' else 1 for kind, held in FACTORED_KINDS.items()
            },
        }
        spec['gaps'] = 1
        # Tᵀ D⁻¹ P of the data under a low-rank noise model, summed over the tiles of each
        # observation, and K of the observation, from its noise model
        spec['modes'] = ('segment_sum', 'segment')
        kernel = self._tile_kernel(self._gap_splines[bucket_index])
        out, resolved = map_reduce(kernel, ensemble, spec, sky)
        return out['hits'], out['map'], self._bucket_model(bucket, out, resolved)

    def _tile_reader(
        self, bucket: TileBucket, reader: ObservationReader[T]
    ) -> tuple[TileReader, Callable[[int], tuple[Any, Any, np.ndarray]]]:
        """A fresh pass of `reader` over the tiles of `bucket`.

        Returns the tile reader, and the read of each tile, filler for the padding tiles.
        """
        reader.reset_failures()  # fresh pass: drop failures recorded by any previous read
        # the reader reads the observations of the bucket in the order of their indices
        position = {int(o): k for k, o in enumerate(bucket.observations)}
        tile_counts = [0] * bucket.n_observations
        for segment, o in enumerate(bucket.segment_observations):
            if o >= 0:
                tile_counts[position[int(o)]] = len(bucket.tiles_of(segment))
        tile_reader = TileReader(reader, bucket.tile_size, tile_counts)

        def read_tile(tile: int) -> tuple[Any, Any, np.ndarray]:
            if not bucket.valid[tile]:
                return tile_reader.filler
            observation = position[int(bucket.segment_observations[bucket.owner[tile]])]
            return tile_reader.read(observation, int(bucket.tile_starts[tile]))

        return tile_reader, read_tile

    def _residual_statistics(
        self,
        acc: AccumulatedModel,
        sky: StokesType,
        blocks: list[list[PyTree[Array]]],
        diagnostics: DiagnosticsConfig,
    ) -> tuple[dict[str, np.ndarray], StokesType | None]:
        r"""One more pass over the data: the residual statistics of every tile, and its binned map.

        Args:
            acc: The accumulated system.
            sky: The map solved for, on the whole sky.
            blocks: The amplitudes solved for, as the system orders them (see
                [`AccumulatedModel.build_system`][]); empty without.
            diagnostics: The statistics to compute.

        Returns:
            The `good` samples of each detector and those of its `chi2` and band `bands`, with their
            expected `band_samples`, turned on, per observation in observation order, `([band,]
            obs, stream, det)`, zero-padded; and the residual binned on the whole sky, before its
            normalization, if turned on. Under a low-rank noise model, the correlated modes are
            marginalized from the $\chi^2$ once the tiles of each observation are summed.
        """
        solved = dict(zip(acc.amplitude_kinds, blocks, strict=True))
        layout = self.placement.layout
        shared_sky = None
        if (scanned := self._scanned_sky) is not None:
            shared_sky = self.placement.broadcast(scanned)
        solution = self.placement.broadcast(sky)
        names = ['good', *(n for n in ('chi2', 'bands') if getattr(diagnostics, n))]
        if diagnostics.bands:
            names.append('band_samples')
        statistics_on = diagnostics.chi2 or bool(diagnostics.bands)
        per_bucket: dict[str, list[np.ndarray]] = {name: [] for name in names}
        residual_map = self.landscape.zeros() if diagnostics.residual_map else None
        for b, reader in enumerate(self.readers):
            bucket = self.layout.buckets[b]
            tile_reader, read_tile = self._tile_reader(bucket, reader)
            # an empty tuple for the kinds absent: the strides of the engine count None as a leaf
            amplitudes = {
                kind: () if kind not in solved else self.placement.gather(solved[kind][b])
                for kind in ('explicit', 'gaps', 'common')
            }

            def read(tile: int, amplitudes=amplitudes, read_tile=read_tile, bucket=bucket) -> Any:
                segment = bucket.owner[tile]

                def at(index):
                    return lambda a: a[index]

                return (
                    read_tile(tile),
                    jax.tree.map(at(tile), amplitudes['explicit']),
                    jax.tree.map(at(tile), amplitudes['gaps']),
                    jax.tree.map(at(segment), amplitudes['common']),
                )

            structures = tuple(
                jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape[1:], a.dtype), amplitudes[k])
                for k in ('explicit', 'gaps', 'common')
            )
            ensemble = HostEnsemble(
                bucket.n_tiles,
                read,
                (tile_reader.structure, *structures),
                policy=Policy(axis=self.placement.axis),
                prefetch=self._prefetches_reads,
                strides=(tile_reader.strides, *(jax.tree.map(lambda _: 1, s) for s in structures)),
                owner=bucket.owner,
                segments=bucket.n_segments,
            )
            kernel = self._residual_kernel(self._gap_splines[b])
            bm = acc.buckets[b]
            modes = statistics_on and bm.modes_core is not None
            spec: dict[str, Any] = dict.fromkeys(names, 1)
            if residual_map is not None:
                spec['map'] = 'sum'
            if modes:
                spec |= {'modes': 'segment_sum', 'low': 1}
            out, _ = map_reduce(kernel, ensemble, spec, shared_sky, solution)
            if residual_map is not None:
                residual_map = furax.tree.add(residual_map, out['map'])
            if modes:
                out = self._marginalize_modes(bm, out, diagnostics)
            for name, arrays in per_bucket.items():
                arrays.append(self.placement.gather(out[name]))
        # each tile is (stream, det), after the band axis, if any
        statistics = {
            name: layout.to_observation_order(
                arrays, detector_axis=3 if name in ('bands', 'band_samples') else 2
            )
            for name, arrays in per_bucket.items()
        }
        for name in ('bands', 'band_samples'):
            if name in statistics:
                statistics[name] = np.moveaxis(statistics[name], 1, 0)
        return statistics, residual_map

    def _marginalize_modes(
        self, bm: BucketModel, out: dict[str, Any], diagnostics: DiagnosticsConfig
    ) -> dict[str, Any]:
        """The statistics of a bucket under a low-rank noise model, its modes marginalized.

        `out` holds those under the diagonal weight, and the sums and low-frequency coefficients
        of the residual that give their correction (see [`modes_correction`][]).
        """
        template = bm.tiles.template.model
        noise = template.noise_model
        assert isinstance(noise, LowRankNoiseModel) and bm.modes_core is not None
        n_samples = template.tod_format.n_samples
        z = jax.vmap(lambda core, u: core.apply(u, noise.edges, noise.sample_rate, n_samples))(
            bm.modes_core, out['modes']
        )

        def apply(tile, _whole, low, z_o):
            return None, modes_correction(tile.model, low, z_o, diagnostics), None

        _, correction, _ = bucket_apply(bm, apply, None, out['low'], z)
        return out | {name: out[name] - value for name, value in correction.items()}

    def _bucket_model(
        self, bucket: TileBucket, out: dict[str, Any], resolved: dict[str, Any]
    ) -> BucketModel:
        """The model of a bucket, from the outputs of its pass over the data and their kinds."""
        strides = _strides(out, resolved)
        templates = template_strides = None
        if self.fit_config.use_templates:
            inverses, inverse_strides = self._factor_common_grams(out['grams'])
            templates = ObservationTemplates.from_inverses(
                out['implicit'], out['operators'], out['inverses'] | inverses
            )
            template_strides = ObservationTemplates.from_inverses(
                strides['implicit'], strides['operators'], strides['inverses'] | inverse_strides
            )
        tile = Tile(out['model'], templates, out['rhs']['explicit'], *out['gaps'])
        tile_strides = Tile(
            strides['model'], template_strides, strides['rhs']['explicit'], *strides['gaps']
        )
        tiles = StackedEnsemble(
            bucket.n_tiles,
            tile,
            tuple(jax.tree.structure(tile).flatten_up_to(tile_strides)),
            policy=Policy(axis=self.placement.axis),
            owner=self.placement.shard(bucket.owner),
            segments=bucket.n_segments,
        )
        modes_data, modes_core = out['modes']
        return BucketModel(tiles, out['rhs']['common'], modes_data, modes_core)

    def _factor_common_grams(
        self, grams: dict[str, tuple[Any, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """The inverses of the Grams summed over the tiles of each observation, and their strides.

        Each observation's Gram is factored on the device that holds it, one per round.
        """
        factored_gram = self._factor
        inverses, strides = {}, {}
        for kind, (gram, coarse) in grams.items():
            if gram is None:
                inverses[kind] = strides[kind] = None
                continue
            n_segments = jax.tree.leaves(gram)[0].shape[0]
            ensemble = StackedEnsemble(
                n_segments, (gram, coarse), policy=Policy(axis=self.placement.axis)
            )

            def factored(member):
                gram, coarse = member
                return factored_gram(replace(gram, coarse=coarse))

            inverse, kinds = map_reduce(factored, ensemble, 'auto')
            inverses[kind] = inverse
            # one per observation: per segment among the tiles
            strides[kind] = _strides(inverse, kinds, segment=True)
        return inverses, strides

    @cached_property
    def _factor(self) -> Callable[[Any], Any]:
        """`factor` of a Gram, with the run's template regularization."""
        templates = self.fit_config.templates
        return partial(
            factor, regularization=0.0 if templates is None else templates.regularization
        )

    @cached_property
    def _prefetches_reads(self) -> bool:
        """Whether to read the next observations while the devices accumulate the current ones.

        Only where the devices have memory of their own: on a CPU device, the reads would hold the
        next TOD in the same memory as the current one's working set, and compete with it for
        the cores.
        """
        return self.placement.local_devices[0].platform != 'cpu'

    def _tile_kernel(
        self, gaps: GapSplines | None = None
    ) -> Callable[[tuple[Any, Any, Array], StokesType | None], dict[str, Any]]:
        """Accumulate one detector tile: its hit map, RHS and model.

        The member is one tile's read, the triple `(observation, padding, valid)` restricted to
        the tile's detectors; the sky to inject, if any, is a shared input that every member
        receives whole. `gaps` is the spline model of the explicit gaps of the bucket, if any.
        """
        return self._tile_kernels(gaps)[0]

    def _residual_kernel(
        self, gaps: GapSplines | None = None
    ) -> Callable[[tuple[Any, ...], StokesType | None, StokesType], dict[str, Any]]:
        """The residual statistics of one detector tile (see [`tile_statistics`][]).

        The member is one tile's read, then its own explicit and gap amplitudes and its
        observation's shared ones, any of them `None`; the sky to inject, if any, and the map
        solved for, on the whole sky, are shared inputs.
        """
        return self._tile_kernels(gaps)[1]

    def _tile_kernels(
        self, gaps: GapSplines | None
    ) -> tuple[Callable[..., Any], Callable[..., Any]]:
        """The kernels of the pass that accumulates the system, and of the residual pass."""
        config = self.fit_config
        noise_sample_rate = None
        if config.weighting.mode == WeightingMode.LOW_RANK:
            noise_sample_rate = self._probed_sample_rate
        landscape = self.landscape
        fill_gaps = config.gaps.treatment == GapTreatment.FILL and not config.binned
        build_templates = config.use_templates
        sky_landscape = landscape.with_stokes(config.read_stokes)
        replaces = config.injection is not None and config.injection.mode == InjectionMode.REPLACE
        factored = self._factor
        diagnostics = config.diagnostics_config or DiagnosticsConfig()

        def cast(tod):
            # The TOD is read in float32, whatever the pipeline's dtype
            return jax.tree.map(lambda x: x.astype(config.dtype), tod)

        def cast_tod(data):
            return {**data, ReaderField.SAMPLE_DATA: cast(data[ReaderField.SAMPLE_DATA])}

        def prepare(data, padding, sky):
            data = cast_tod(data)
            if sky is not None:  # the samples to inject into, per detector
                shape = jax.tree.leaves(data[ReaderField.SAMPLE_DATA])[0].shape
                H = acquisition_operator(data, config, sky_landscape)
                data = inject(H, sky, data, unpadded(padding, shape), replaces)
            return data

        def kernel(member, sky):
            # Gap filling needs whole observations, so it only runs with one tile per observation
            # (see `MapMakingConfig`).
            data, padding, use = member
            data = prepare(data, padding, sky)
            tod = raw = data[ReaderField.SAMPLE_DATA]
            if fill_gaps:
                # The RHS uses the filled TOD; the models are still built from the raw one.
                tod = jax.lax.cond(
                    use,
                    lambda: fill_observation_gaps(data, padding),
                    lambda: raw,  # nothing to fill
                )
            return accumulate(data, tod, padding, use)

        def fill_observation_gaps(data, padding):
            obs = ObservationModel.create(
                data, padding, config, landscape, noise_sample_rate=noise_sample_rate
            )
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

        def residual_kernel(member, sky, solution):
            (data, padding, use), *amplitudes = member
            explicit, gap_amplitudes, common = (
                None if not jax.tree.leaves(a) else a for a in amplitudes
            )
            data = prepare(data, padding, sky)
            data, tod, obs, built = build(data, data[ReaderField.SAMPLE_DATA], padding, use)
            return tile_statistics(
                obs, built, tod, solution, explicit, common, gap_amplitudes, diagnostics
            )

        def build(data, tod, padding, use):
            # the tile's data and TOD restricted to the streams of the map, its model, and its
            # templates, if any
            temperature = None
            if config.read_stokes != landscape.stokes:
                # The extra streams read, the I one for the T2P template, stay out of the model
                temperature = data[ReaderField.SAMPLE_DATA].i
                data, tod = restrict_streams(data, tod, landscape.stokes)
            # Padding/failed observations contribute nothing
            obs = ObservationModel.create(
                data,
                padding,
                config,
                landscape,
                use,
                gaps,
                noise_sample_rate=noise_sample_rate,
                temperature=temperature,
            )
            built = None
            if build_templates:  # templates are rejected with gap filling (see `MapMakingConfig`)
                built = ObservationTemplates.assemble(data, config, obs, tod, temperature)
            return data, tod, obs, built

        def accumulate(data, tod, padding, use):
            data, tod, obs, built = build(data, tod, padding, use)

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

            if built is not None:
                wd = built.weighted_tod
                rhs_i, implicit = obs.H.T(wd), built.implicit
                operators, grams = built.operators, built.grams
            else:
                # Gaps filled: skip the data-side mask so the fill survives N⁻¹.
                rhs_i = (obs.rhs_operator_prefilled if fill_gaps else obs.rhs_operator)(tod)
                implicit, wd = None, None
                operators = grams = dict.fromkeys(FACTORED_KINDS)
            return {
                'hits': hits_i,
                'map': rhs_i,
                'gaps': (None, None) if obs.gaps is None else gap_system(obs, tod),
                'model': obs,
                'implicit': implicit,
                'operators': operators,
                'inverses': {
                    kind: None if gram is None else factored(gram)
                    for kind, gram in grams.items()
                    if FACTORED_KINDS[kind] == 'stack'
                },
                'grams': {
                    kind: (None, None)
                    if gram is None
                    else (replace(gram, coarse=None), gram.coarse)
                    for kind, gram in grams.items()
                    if FACTORED_KINDS[kind] == 'sum'
                },
                # the tile's share of the right-hand side of each kind of amplitudes
                'rhs': {kind: None if op is None else op.T(wd) for kind, op in operators.items()},
                'modes': modes_system(obs, tod),
            }

        def modes_system(obs, tod):
            # the tile's share of Tᵀ D⁻¹ P d, P the mask, whose correction the system subtracts
            # from its right-hand side, and K of the observation
            if not isinstance(model := obs.noise_model, LowRankNoiseModel):
                return None, None
            assert isinstance(obs.W, WeightOperator)
            modes = model.modes_operator(obs.tod_format.structure)
            data = modes.T(obs.W.weight(obs.M(tod)))
            return data, model.modes_core(obs.tod_format.n_samples)

        def gap_system(obs, tod):
            # the right-hand side Cᵀ N⁻¹ M d of the gap amplitudes, and the inverse of the diagonal
            # of their white-noise Gram Cᵀ D C, which preconditions them
            assert isinstance(obs.W, WeightOperator) and obs.gaps is not None
            C = obs.gap_basis()
            rhs = C.T(obs.W.weight(obs.M(tod)))
            white = obs.diag_W().weight
            diagonal = C.squared().T(white(furax.tree.ones_like(obs.tod_format.structure)))
            usable = diagonal > 0  # the slots past the amplitudes of a detector are unused
            return rhs, jnp.where(usable, 1 / jnp.where(usable, diagonal, 1), 1)

        return kernel, residual_kernel

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


def _detector_statistics(
    statistics: dict[str, np.ndarray] | None, diagnostics: DiagnosticsConfig
) -> dict[str, np.ndarray] | None:
    r"""The reduced $\chi^2$ of each detector, overall and per band, `(obs, stream, det)`."""
    if statistics is None or len(statistics) == 1:  # the good samples alone
        return None
    good = statistics['good']
    usable = good > 0
    columns = {}
    if 'chi2' in statistics:
        columns['chi2'] = np.where(usable, statistics['chi2'] / np.where(usable, good, 1), np.nan)
    edges = [0.0, *diagnostics.bands] if 'bands' in statistics else []
    for k, low in enumerate(edges):
        high = f'{edges[k + 1]:g}' if k + 1 < len(edges) else 'nyquist'
        samples = statistics['band_samples'][k]
        reduced = statistics['bands'][k] / np.where(samples > 0, samples, 1)
        columns[f'chi2_{low:g}_{high}hz'] = np.where(usable & (samples > 0), reduced, np.nan)
    return columns


def _excluded_message(failures: list[tuple[str, str]], limit: int = 5) -> str:
    """The warning for the observations that were excluded, with the reason of each.

    Args:
        failures: The name and the reason of each excluded observation.
        limit: The number of observations listed; the others are counted.
    """
    lines = [f'Excluded {len(failures)} observation(s):']
    lines += [f'  {name}: {reason}' for name, reason in failures[:limit]]
    if len(failures) > limit:
        lines.append(f'  and {len(failures) - limit} more')
    return '\n'.join(lines)


def _strides(outputs: PyTree[Any], resolved: PyTree[Any], segment: bool = False) -> PyTree[Any]:
    """The stride of each output leaf of `map_reduce` in an ensemble of tiles.

    An output stacked per tile has stride 1, one kept or summed per segment the stride
    `'segment'`, and one shared by the bucket None. With `segment`, the outputs are those of an
    ensemble of segments, whose stacked outputs are per segment among the tiles.

    The resolved spec puts `None` on the shared leaves, which `jax.tree.map` over the outputs
    would read as a missing subtree; the returned tree has the structure of the outputs, so it
    maps with them.
    """
    treedef = jax.tree.structure(outputs)
    kinds = treedef.flatten_up_to(resolved)
    per_member = 'segment' if segment else 1
    strides = [
        None if kind is None else 'segment' if kind in ('segment', 'segment_sum') else per_member
        for kind in kinds
    ]
    return jax.tree.unflatten(treedef, strides)
