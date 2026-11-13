import logging
import time
from collections.abc import Collection, Sequence
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils as mhu
from jaxtyping import Array, Bool, Float, Int64, Integer

import furax.linalg
import furax.tree
from furax.core import HostEnsemble, IndexOperator, Policy, StackedEnsemble, map_reduce
from furax.obs.landscapes import HealpixLandscape, StokesLandscape, WCSLandscape
from furax.obs.stokes import StokesType
from furax.profiling import format_bytes

from . import _report
from ._accumulated import AccumulatedModel
from ._bucket import AmplitudeKind
from ._catalog import ObservationCatalog, ProbeRecord, Read, detector_table, gather_rows
from ._gaps import GapSplines, gap_budget, knot_spacing, max_gap_amplitudes
from ._injection import load_sky
from ._landscape import explicit_wcs_geometry, scan_wcs_footprint
from ._model import TileModel
from ._observation import (
    AbstractLazyObservation,
    ObservationBufferShape,
    ObservationProbe,
    ReaderField,
)
from ._pass import TILE_REDUCTIONS, bucket_tiles, tile_kernel
from ._provenance import provenance
from ._reader import ObservationReader, TileReader, detector_tile_structure
from .config import GapTreatment, InjectionMode, MapMakingConfig, NoiseSource, WeightingMode
from .layout import TileLayout, TilePlacement
from .results import CONFIG_FILE, PROVENANCE_FILE, MapMakingResults, write_json
from .templates import required_reader_fields

logger = logging.getLogger(__name__)


class MultiObservationMapMaker[T]:
    """Class for mapping multiple observations together."""

    def __init__(
        self,
        observations: Sequence[AbstractLazyObservation[T]],
        config: MapMakingConfig | None = None,
    ) -> None:
        self.observations = observations
        self.config = config or MapMakingConfig()
        # Build the landscape and readers up front, outside a mesh context
        self.landscape = self.build_landscape()
        self.readers = self.build_readers(self.reader_fields)

    def run(self, out_dir: str | Path, resume: bool = False) -> MapMakingResults:
        """Runs the mapmaker and return results after saving them to the given directory.

        The configuration and the provenance are written to `out_dir` before the maps are
        computed.

        Args:
            out_dir: Directory of the products, and of the checkpoints of the solves.
            resume: Resume the solves from the checkpoints in `out_dir`, if any.
        """
        # Rank 0 alone saves the configuration and run provenance (versions etc.)
        is_writer = jax.process_index() == 0
        out_dir = Path(out_dir).resolve()
        if is_writer:
            out_dir.mkdir(parents=True, exist_ok=True)
            self.config.dump_yaml(out_dir / CONFIG_FILE)
            write_json(out_dir / PROVENANCE_FILE, provenance())
            logger.info(f'Saved mapmaking configuration and provenance to {out_dir}')

        self._log_layout()
        results = self.make_maps(checkpoint_dir=out_dir, resume=resume)

        # Rank 0 saves the replicated outputs
        if is_writer:
            results.save(out_dir)
            logger.info(f'Saved results to {out_dir}')

        # Barrier so other ranks don't race ahead while rank 0 is still writing.
        mhu.sync_global_devices('mapmaker.run.save_done')

        return results

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
        # Loaded before the passes over the data, so that a bad injection fails early
        injected_sky = self._load_injected_sky()
        is_response = (
            self.config.injection is not None
            and self.config.injection.mode == InjectionMode.RESPONSE
        )

        with jax.set_mesh(self.placement.mesh):
            # Single read pass, so each observation is read/preprocessed exactly once.
            start = time.perf_counter()
            acc = self.build_model_and_accumulate(None if is_response else injected_sky)
            jax.block_until_ready(acc)
            logger.info(f'Finished the pass over the data in {time.perf_counter() - start:.0f} s')

            self._catalog.reads = self._gather_reads()
            if failures := self._catalog.failures():
                # The catalog is the same on every process: one warning is enough
                level = logging.WARNING if jax.process_index() == 0 else logging.DEBUG
                logger.log(level, _excluded_message(failures))
            drops = _report.detector_drops(acc.buckets, self.placement)
            if n_dropped := int(drops['over_flagged'].sum()):
                logger.info(
                    f'Dropped {n_dropped} detectors flagged above '
                    f'{self.config.gaps.max_flagged_fraction:.0%} '
                    f'in {int(drops["over_flagged"].any(axis=-1).sum())} observations'
                )
            if n_over := int(drops['over_gap_budget'].sum()):
                # the budget is meant to fit every detector kept: this one did not
                logger.warning(
                    f'Dropped {n_over} detectors needing more gap amplitudes than the budget '
                    f'in {int(drops["over_gap_budget"].any(axis=-1).sum())} observations'
                )
            noise_parameters = None
            if self.config.weighting.mode != WeightingMode.IDENTITY:
                noise_parameters = _report.noise_parameters(acc.buckets, self.placement)

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

            system = acc.build_system(S, BJ, injected_sky if is_response else None)
            if (two_level := self.config.solver.two_level) is not None:
                system = system.deflate(two_level)
            solved = system.solve(
                self.config.solver,
                self.placement,
                None if checkpoint_dir is None else Path(checkpoint_dir),
                resume,
            )
            sky = solved.x['map']
            amplitudes = _report.template_amplitudes(acc.buckets, solved.x, self.placement) or None

        # Built only now, so that their copies are not held through the solve
        return MapMakingResults(
            map=S.T(sky),  # all sky pixels including those not estimated (zero)
            icov=jnp.moveaxis(icov, [-2, -1], [0, 1]),  # (*pixels, ns, ns) → (ns, ns, *pixels)
            hit_map=jnp.where(valid_pixels, acc.hit_map, 0),  # excluded pixels have zero hits
            solver_stats=solved.solver_stats,
            landscape=self.landscape,
            template_amplitudes=amplitudes,
            injected_map=None if solved.injected_sky is None else S.T(solved.injected_sky),
            observations=self._catalog.to_table(drops),
            # the noise model covers the streams of the model's TOD, not all those read
            detectors=detector_table(
                self._catalog.reads,
                noise_parameters,
                acc.buckets[0].template.model.tod_format.stokes,
                drops,
            ),
        )

    def build_model_and_accumulate(self, scanned_sky: StokesType | None = None) -> AccumulatedModel:
        """Build the model and accumulate the hit map and right-hand side in one pass over the data.

        Padding tiles and failed observations are gated out using a zero-mask, so they drop from
        the system.

        Args:
            scanned_sky: The simulated sky to scan into the TOD, if any.

        This method must run under `jax.set_mesh(self.placement.mesh)`.
        """
        hit_map = jnp.zeros(self.landscape.shape, jnp.int64)
        rhs: dict[str, Any] = {'map': self.landscape.zeros()}
        # The sky to scan in is an argument of the kernels, held by every device for this pass
        # only, rather than a constant compiled into them
        sky = None
        if scanned_sky is not None:
            sky = self.placement.broadcast(scanned_sky)
        buckets = []
        for b, reader in enumerate(self.readers):
            hits, map_rhs, bucket_rhs, tiles = self._accumulate_bucket(b, reader, sky)
            hit_map = hit_map + hits
            rhs['map'] = furax.tree.add(rhs['map'], map_rhs)
            for kind, kind_rhs in bucket_rhs.items():
                rhs.setdefault(kind, []).append(kind_rhs)
            buckets.append(tiles)
        return AccumulatedModel(buckets=buckets, hit_map=hit_map, rhs=rhs)

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

    def build_landscape(self) -> StokesLandscape:
        """Build the sky grid a run solves on.

        The grid comes from the config alone for HEALPix and for a WCS with an explicit geometry. For
        a WCS without one, the observations are scanned for their combined footprint.
        """
        landscape = self.config.landscape
        stokes, dtype = landscape.stokes, self.config.dtype
        if (hp := landscape.healpix) is not None:
            return HealpixLandscape(
                nside=hp.nside,
                stokes=stokes,
                dtype=dtype,
                nested=hp.ordering == 'nest',
            )
        assert landscape.wcs is not None  # LandscapeConfig sets exactly one of healpix and wcs
        if landscape.wcs.has_geometry:
            shape, wcs = explicit_wcs_geometry(landscape.wcs)
        else:
            shape, wcs = scan_wcs_footprint(self.observations, landscape.wcs)
        return WCSLandscape.from_wcs(shape, wcs, stokes, dtype)

    def build_readers(self, required_fields: Collection[str]) -> list[ObservationReader[T]]:
        """Build one reader per bucket, over that bucket's observations only.

        A reader reads its bucket's observations in the order of their indices, into buffers of
        whole tiles sized by exactly the observations it can be asked to read.
        """
        shapes = self._buffer_shapes
        failed = self._catalog.probe_failed
        return [
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
        ]

    @property
    def reader_fields(self) -> set[str]:
        rhs_fields = {ReaderField.METADATA, ReaderField.SAMPLE_DATA}
        model_fields = TileModel.required_reader_fields(self.config)
        templates = self.config.templates
        template_fields = set() if templates is None else required_reader_fields(templates)
        return model_fields | rhs_fields | template_fields

    @property
    def n_observations(self) -> int:
        """Total number of observations across all processes."""
        return len(self.observations)

    @cached_property
    def layout(self) -> TileLayout:
        """How the observations are bucketed, split into tiles and laid out over the devices."""
        return TileLayout.create(
            self._buffer_shapes,
            n_devices=jax.device_count(),
            max_buckets=self.config.max_buckets,
            detector_tile_size=self.config.detector_tile_size,
        )

    @cached_property
    def placement(self) -> TilePlacement:
        """The tiles of the layout, placed on the devices of the job."""
        return TilePlacement.create(self.layout)

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

    def _load_injected_sky(self) -> StokesType | None:
        """Load the simulated sky, if one is injected.

        A sky scanned into the TOD has the Stokes components that are read; one solved for as a
        response, those of the map.
        """
        if (injection := self.config.injection) is None:
            return None
        if injection.mode == InjectionMode.RESPONSE:
            return load_sky(injection, self.landscape)
        return load_sky(injection, self.landscape.with_stokes(self.config.read_stokes))

    def _accumulate_bucket(
        self, bucket_index: int, reader: ObservationReader[T], sky: StokesType | None
    ) -> tuple[Int64[Array, '...'], StokesType, dict[AmplitudeKind, Any], StackedEnsemble]:
        """One bucket's pass over the data: its hit map, its right-hand sides and its tiles.

        The pass is an engine call on a host ensemble of the bucket's tiles: each round launches
        one tile on each device while its process reads the next round. An observation is read
        once, when its first tile is, and held on the host until its last tile has been read, so
        only one observation per device is in memory at a time, and the processes need not keep
        in step until the device sum at the end. The stacked outputs land sharded over the tiles
        and the segments, which is the layout of the bucket's tiles.
        """
        reader.reset_failures()  # fresh pass: drop failures recorded by any previous read
        bucket = self.layout.buckets[bucket_index]
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
        kernel = tile_kernel(self.fit_config, self.landscape, self._gap_splines[bucket_index])
        (hits, map_rhs, rhs, shared, tile), resolved = map_reduce(
            kernel, ensemble, TILE_REDUCTIONS, sky
        )
        tiles = bucket_tiles(shared, tile, resolved, bucket, self.placement, self.fit_config)
        return hits, map_rhs, {kind: v for kind, v in rhs.items() if v is not None}, tiles

    @cached_property
    def _prefetches_reads(self) -> bool:
        """Whether to read the next observations while the devices accumulate the current ones.

        Only where the devices have memory of their own: on a CPU device, the reads would hold the
        next TOD in the same memory as the current one's working set, and compete with it for
        the cores.
        """
        return self.placement.local_devices[0].platform != 'cpu'

    @cached_property
    def _gap_splines(self) -> list[GapSplines | None]:
        """The spline model of the explicit gaps of each bucket; `None` without.

        The knot spacing of a bucket, in samples, is `ExplicitGapsConfig.resolution` at the sample
        rates its observations probed, the finest if they differ. Its budget of amplitudes fits
        every observation in it: the most amplitudes their probed good ranges need, when they
        all reported them at that spacing, or else the bound that holds for any flags (see
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
            read = {i: p for i in bucket.observations if isinstance(p := probes[i], ProbeRecord)}
            spacings = [self._knot_spacing(i, p) for i, p in read.items()]
            spacing = min(spacings, default=1)
            counts = [p.gap_amplitudes for p in read.values() if p.gap_amplitudes is not None]
            if read and len(counts) == len(read) and set(spacings) == {spacing}:
                budget = max(counts)
            else:
                budget = max(
                    gap_budget(shapes[i].sample_count, gaps.max_flagged_fraction)
                    for i in bucket.observations
                )
            budget = max(budget, 1)  # the basis indexes its slots: an empty budget has none
            logger.info(
                f'Bucket {b}: {budget} gap amplitudes per detector, knots every {spacing} samples'
            )
            models.append(GapSplines(spacing, budget))
        return models

    def _knot_spacing(self, i: int, probe: ProbeRecord) -> int:
        """The knot spacing of the explicit gaps of observation `i`, in samples."""
        if probe.knot_spacing is None:
            msg = (
                f'Observation {self._catalog.names[i]}: explicit gaps need its sample rate, '
                'which its probe does not report'
            )
            raise ValueError(msg)
        return probe.knot_spacing

    @cached_property
    def _buffer_shapes(self) -> list[ObservationBufferShape]:
        """The shapes the buckets and the readers size their buffers from."""
        if not self.config.fft_friendly_sample_count:
            return self._probe_shapes
        return [shape.fft_friendly() for shape in self._probe_shapes]

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
    def _catalog(self) -> ObservationCatalog:
        """What the run knows of every observation, starting from its probe.

        Probing costs an open, so the observations are dealt round-robin to the processes and the
        probes are gathered. A probe that raises must not crash the process (it would deadlock the
        others at the gather), so its exception is kept instead.
        """
        n_proc = jax.process_count()
        need_intervals = ReaderField.SCANNING_INTERVALS in self.reader_fields
        explicit = self._explicit_gaps
        # the good ranges size the gap amplitudes (`ExplicitGapsConfig.probe`)
        count_gaps = explicit and self.config.gaps.explicit.probe
        local: dict[int, ProbeRecord | str] = {}
        for i in range(jax.process_index(), self.n_observations, n_proc):
            try:
                probe = self.observations[i].probe(need_intervals, explicit, count_gaps)
                local[i] = self._probe_record(probe)
            except Exception as error:
                logger.exception('Probe of observation %d failed', i)
                local[i] = f'{type(error).__name__}: {error}'
        names = [observation.name for observation in self.observations]
        return ObservationCatalog(names, gather_rows(local))

    def _probe_record(self, probe: ObservationProbe) -> ProbeRecord:
        """What the run keeps of a probe: its shape, and what the explicit gaps derive from it.

        The knot spacing follows from the probed sample rate. With `ExplicitGapsConfig.probe`, the
        amplitudes the probed good ranges need at that spacing are counted.
        """
        spacing = count = None
        if self._explicit_gaps and probe.sample_rate is not None:
            gaps = self.config.gaps
            spacing = knot_spacing(gaps.explicit.resolution, probe.sample_rate)
            if gaps.explicit.probe and probe.good_ranges is not None:
                assert gaps.max_flagged_fraction is not None  # required by explicit gaps
                count = max_gap_amplitudes(probe.good_ranges, spacing, gaps.max_flagged_fraction)
        return ProbeRecord(probe.shape, spacing, count)

    @property
    def _explicit_gaps(self) -> bool:
        """Whether the gaps are explicit, their amplitudes solved for with the map."""
        return self.config.gap_treatment == GapTreatment.EXPLICIT


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
