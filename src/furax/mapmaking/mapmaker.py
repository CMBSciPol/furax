import logging
import math
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
from jaxtyping import Array, Bool, Float, Integer

import furax.linalg
import furax.tree
from furax.core import IndexOperator, StackedEnsemble, lift, map_reduce
from furax.obs.landscapes import (
    HealpixLandscape,
    LocalStokesLandscape,
    StokesLandscape,
    WCSLandscape,
)
from furax.obs.stokes import StokesType
from furax.profiling import format_bytes

from . import _report
from ._bucket import block_jacobi, bucket_rhs, modes_core, selected
from ._catalog import (
    ObservationCatalog,
    ProbeRecord,
    Read,
    detector_table,
    gather_rows,
    observation_chi2,
)
from ._diagnostics import modes_statistics, statistics_correction
from ._gaps import GapSplines, bridge_amplitudes, gap_budget, knot_spacing, max_gap_amplitudes
from ._injection import load_sky
from ._landscape import explicit_wcs_geometry, scan_wcs_footprint
from ._observation import (
    AbstractLazyObservation,
    ObservationBufferShape,
    ObservationProbe,
    ReaderField,
)
from ._pass import (
    PASS_REDUCTIONS,
    PassTotals,
    bucket_tiles,
    fit_kernel,
    fit_low_rank,
    pass_kernel,
    residual_kernel,
)
from ._provenance import provenance
from ._reader import ObservationReader, detector_tile_structure, tile_reads
from ._solve import MapMakingSystem
from ._tile import AMPLITUDE_STRIDES, Tile
from .config import (
    DiagnosticsConfig,
    GapTreatment,
    InjectionMode,
    MapMakingConfig,
    NoiseSource,
    WeightingMode,
)
from .layout import TileBucket, TileLayout, TilePlacement
from .noise import LowRankNoiseModel, ModesCore, bin_edges
from .results import CONFIG_FILE, PROVENANCE_FILE, MapMakingResults, write_json
from .templates import required_reader_fields

logger = logging.getLogger(__name__)

_SAMPLE_RATE_TOLERANCE = 1e-3
"""Relative spread of the observation sample rates the low-rank weighting accepts as one rate."""


class MultiObservationMapMaker:
    """Maps observations together under one configuration.

    A mapmaker is defined by its configuration alone: the same instance can make maps of
    different datasets through successive calls of [`run`][furax.mapmaking.MultiObservationMapMaker.run]
    or [`make_maps`][furax.mapmaking.MultiObservationMapMaker.make_maps].

    Examples:
        >>> maker = MultiObservationMapMaker(config)
        >>> maker.run(observations, 'maps/season1')
        >>> maker.run(other_observations, 'maps/season2')
    """

    def __init__(self, config: MapMakingConfig | None = None) -> None:
        self.config = config or MapMakingConfig()

    def run[T](
        self,
        observations: Sequence[AbstractLazyObservation[T]],
        out_dir: str | Path,
        resume: bool = False,
    ) -> MapMakingResults:
        """Maps the observations and returns the results after saving them to the given directory.

        The configuration and the provenance are written to `out_dir` before the maps are
        computed.

        Args:
            observations: The observations to map together.
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

        prepared = self.prepare(observations)
        prepared._log_layout()
        results = prepared.make_maps(checkpoint_dir=out_dir, resume=resume)

        # Rank 0 saves the replicated outputs
        if is_writer:
            results.save(out_dir)
            logger.info(f'Saved results to {out_dir}')

        # Barrier so other ranks don't race ahead while rank 0 is still writing.
        mhu.sync_global_devices('mapmaker.run.save_done')

        return results

    def make_maps[T](
        self,
        observations: Sequence[AbstractLazyObservation[T]],
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
    ) -> MapMakingResults:
        """Computes the maps and other products of the observations, without saving them.

        See [`MapMakingRun.make_maps`][furax.mapmaking.MapMakingRun.make_maps] for the arguments.
        """
        return self.prepare(observations).make_maps(checkpoint_dir, resume)

    def prepare[T](self, observations: Sequence[AbstractLazyObservation[T]]) -> 'MapMakingRun[T]':
        """Probes the observations and lays them out over the devices, ready to be mapped."""
        return MapMakingRun(observations, self.config)


class MapMakingRun[T]:
    """One dataset under one configuration: probed, laid out over the devices, ready to solve.

    Built by [`MultiObservationMapMaker.prepare`][furax.mapmaking.MultiObservationMapMaker.prepare].
    """

    def __init__(
        self,
        observations: Sequence[AbstractLazyObservation[T]],
        config: MapMakingConfig,
    ) -> None:
        self.observations = observations
        self.config = config
        # Build the landscape and readers up front, outside a mesh context
        self.landscape = self.build_landscape()
        self.readers = self.build_readers(self.reader_fields)

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
            buckets, totals = self.read_data(None if is_response else injected_sky)
            jax.block_until_ready((buckets, totals))
            logger.info(f'Finished the pass over the data in {time.perf_counter() - start:.0f} s')

            self._catalog.reads = self._gather_reads()
            if failures := self._catalog.failures():
                # The catalog is the same on every process: one warning is enough
                level = logging.WARNING if jax.process_index() == 0 else logging.DEBUG
                logger.log(level, _excluded_message(failures))
            drops = _report.detector_drops(buckets, self.placement)
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
                noise_parameters = _report.noise_parameters(buckets, self.placement)

            BJ = block_jacobi(buckets, self.config.binned)
            icov = BJ.blocks.block_until_ready()
            logger.info('Computed the block-Jacobi preconditioner')

            valid_pixels = self.pixel_selection(totals.hit_map, icov)
            # Select valid pixels on the (trailing) sky axes, leaving the leading Stokes axis intact.
            S = IndexOperator(
                (..., *jnp.where(valid_pixels)), in_structure=self.landscape.structure
            )
            n_observed = jnp.sum(totals.hit_map > 0)
            logger.info(
                f'Selected {jnp.sum(valid_pixels)} pixels ({n_observed} seen, '
                f'{valid_pixels.size} total)'
            )

            response_sky = injected_sky if is_response else None
            # the tiles of the system read and bin the selected pixels alone
            pixels = LocalStokesLandscape.from_boolean_mask(self.landscape, valid_pixels)
            selected_buckets = [selected(tiles, pixels) for tiles in buckets]
            system = MapMakingSystem.create(selected_buckets, totals.rhs, S, BJ, response_sky)
            if (two_level := self.config.solver.two_level) is not None:
                system = system.deflate(two_level)
            solved = system.solve(
                self.config.solver,
                self.placement,
                None if checkpoint_dir is None else Path(checkpoint_dir),
                resume,
            )
            sky = solved.x['map']
            amplitudes = _report.template_amplitudes(buckets, solved.x, self.placement) or None

            statistics = residual_map = None
            if (diagnostics := self.config.diagnostics_config) is not None:
                start = time.perf_counter()
                scanned_sky = None if is_response else injected_sky
                statistics, residual_rhs = self._residual_statistics(
                    buckets, solved.x | {'map': S.T(sky)}, scanned_sky, diagnostics
                )
                if residual_rhs is not None:
                    # binned under the white weight, the system of the block-Jacobi preconditioner
                    residual_map = S.T((S @ BJ.I @ S.T).reduce()(S(residual_rhs)))
                message = f'Residual pass in {time.perf_counter() - start:.0f} s'
                if 'chi2' in statistics:
                    chi2 = observation_chi2(statistics).sum()
                    good = statistics['good'].sum()
                    message += f': chi2/sample={chi2 / max(good, 1):.4f} over {int(good)} samples'
                logger.info(message)

        # Built only now, so that their copies are not held through the solve
        return MapMakingResults(
            map=S.T(sky),  # all sky pixels including those not estimated (zero)
            icov=jnp.moveaxis(icov, [-2, -1], [0, 1]),  # (*pixels, ns, ns) → (ns, ns, *pixels)
            hit_map=jnp.where(valid_pixels, totals.hit_map, 0),  # excluded pixels have zero hits
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
                buckets[0].template.tod_format.stokes,
                drops,
                None if diagnostics is None else _detector_statistics(statistics, diagnostics),
            ),
        )

    def read_data(
        self, scanned_sky: StokesType | None = None
    ) -> tuple[list[StackedEnsemble], PassTotals]:
        """Read the data once: the tiles of each bucket, and what the pass sums over them.

        The tiles hold the model of each detector tile (see [`Tile`][furax.mapmaking._tile.Tile]).
        The totals are the hit map and the right-hand side of the normal system.

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
        for b in range(len(self.readers)):
            tiles, outputs = self._read_bucket(b, sky)
            if (tod := outputs.pop('tod')) is not None:
                # the correlated modes couple the tiles of each observation: the right-hand side
                # is that of the bucket's TOD, held for this bucket only
                outputs['rhs'] = dict.fromkeys(PASS_REDUCTIONS['rhs']) | bucket_rhs(tiles)(tod)
                del tod
            hit_map = hit_map + outputs['hit_map']
            rhs_of_bucket = outputs['rhs']
            rhs['map'] = furax.tree.add(rhs['map'], rhs_of_bucket.pop('map'))
            for kind, kind_rhs in rhs_of_bucket.items():
                if kind_rhs is not None:  # a kind of amplitudes the run has
                    rhs.setdefault(kind, []).append(kind_rhs)
            # the unfactored shared Gram must not stay on the devices through the next pass
            del outputs
            buckets.append(tiles)
        return buckets, PassTotals(hit_map, rhs)

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
        model_fields = Tile.required_reader_fields(self.config)
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

    def _read_bucket(
        self, bucket_index: int, sky: StokesType | None
    ) -> tuple[StackedEnsemble, dict[str, Any]]:
        """One bucket's pass over the data: its tiles, and the kernel's outputs combined over them.

        The outputs are those of [`pass_kernel`][furax.mapmaking._pass.pass_kernel], combined as
        [`PASS_REDUCTIONS`][furax.mapmaking._pass.PASS_REDUCTIONS] says.

        The pass is an engine call on a host ensemble of the bucket's tiles: each round launches
        one tile on each device while its process reads the next round. An observation is read
        once, when its first tile is, and held on the host until its last tile has been read, so
        only one observation per device is in memory at a time, and the processes need not keep
        in step until the device sum at the end. The stacked outputs land sharded over the tiles
        and the segments, which is the layout of the bucket's tiles.
        """
        reader = self.readers[bucket_index]
        bucket = self.layout.buckets[bucket_index]
        fitted = None
        if self.config.weighting.mode == WeightingMode.LOW_RANK:
            fitted = self._fit_low_rank(bucket, reader, sky)
        reader.reset_failures()  # fresh pass: drop failures recorded by any previous read
        reads = tile_reads(bucket, reader, self.placement.axis, self._prefetches_reads)
        cores = None
        if fitted is not None:
            models, cores = fitted
            # each tile takes its rows of the model of its observation, where the fit left it
            reads = reads.zip((models, _first_detectors(bucket)), strides=('segment', 1))
        kernel = pass_kernel(self.fit_config, self.landscape, self._gap_splines[bucket_index])
        outputs, resolved = map_reduce(kernel, reads, PASS_REDUCTIONS, sky)
        tiles = bucket_tiles(outputs, resolved, bucket, self.placement, self.fit_config, cores)
        return tiles, outputs

    def _residual_statistics(
        self,
        buckets: list[StackedEnsemble],
        x: dict[str, Any],
        scanned_sky: StokesType | None,
        diagnostics: DiagnosticsConfig,
    ) -> tuple[dict[str, np.ndarray], StokesType | None]:
        r"""One more pass over the data: the residual statistics of every tile, and its binned map.

        Args:
            buckets: The tiles of each bucket.
            x: The unknowns solved for, the `'map'` on the whole sky.
            scanned_sky: The simulated sky scanned into the TOD, if any.
            diagnostics: The statistics to compute.

        Returns:
            The `good` samples of each detector and those of its `chi2` and band `bands`, with
            their expected `band_samples`, turned on, per observation in observation order,
            `([band,] obs, stream, det)`, zero-padded; and the residual binned on the whole sky,
            before its normalization, if turned on. Under a low-rank noise model, the $\chi^2$ of
            each detector is that of its residual less the correlated modes, fitted once the tiles
            of each observation are summed, and `modes_chi2` and `modes_samples`, `(obs,)`, are
            the $\chi^2$ of the modes and its number of amplitudes.
        """
        kinds = buckets[0].template.kinds
        shared_sky = None if scanned_sky is None else self.placement.broadcast(scanned_sky)
        # the map every tile reads whole
        solution = {'map': self.placement.broadcast(x['map'])}
        names = ['good', *(n for n in ('chi2', 'bands') if getattr(diagnostics, n))]
        if diagnostics.bands:
            names.append('band_samples')
        statistics_on = diagnostics.chi2 or bool(diagnostics.bands)
        kernel = residual_kernel(self.fit_config, self.landscape, diagnostics)
        per_tile: dict[str, list[np.ndarray]] = {name: [] for name in names}
        # the noise model of the run, low-rank or not, is the same in every bucket
        modes = statistics_on and buckets[0].template.modes_core is not None
        per_segment: dict[str, list[np.ndarray]] = {}
        if modes:
            per_segment = {'modes_chi2': [], 'modes_samples': []}
        residual_map = self.landscape.zeros() if diagnostics.residual_map else None
        for b, (reader, tiles) in enumerate(zip(self.readers, buckets, strict=True)):
            reader.reset_failures()  # fresh pass: drop failures recorded by any previous read
            reads = tile_reads(
                self.layout.buckets[b], reader, self.placement.axis, self._prefetches_reads
            )
            # each tile reads the tile the pass built, and its own amplitudes or its observation's
            amplitudes = {kind: x[kind][b] for kind in kinds}
            strides = jax.tree.leaves(
                {
                    kind: jax.tree.map(lambda _, k=kind: AMPLITUDE_STRIDES[k], x[kind][b])
                    for kind in kinds
                }
            )
            reads = reads.zip((tiles.stacked, amplitudes), strides=(*tiles.strides, *strides))
            spec: dict[str, Any] = dict.fromkeys(names, 1)
            if residual_map is not None:
                spec['map'] = 'sum'
            if modes:
                spec |= {'modes': 'segment_sum', 'low': 1}
            out, _ = map_reduce(kernel, reads, spec, shared_sky, solution)
            if residual_map is not None:
                residual_map = furax.tree.add(residual_map, out['map'])
            if modes:
                out = _marginalized_modes(buckets[b], out, diagnostics)
            for name, arrays in (per_tile | per_segment).items():
                arrays.append(self.placement.gather(out[name]))
        # each tile is ([band,] stream, det)
        layout = self.placement.layout
        statistics = {
            name: layout.to_observation_order(arrays, detector_axis=arrays[0].ndim - 1)
            for name, arrays in per_tile.items()
        } | {name: layout.to_observation_order(arrays) for name, arrays in per_segment.items()}
        for name in ('bands', 'band_samples'):
            if name in statistics:
                statistics[name] = np.moveaxis(statistics[name], 1, 0)
        return statistics, residual_map

    def _fit_low_rank(
        self, bucket: TileBucket, reader: ObservationReader[T], sky: StokesType | None
    ) -> tuple[LowRankNoiseModel, ModesCore]:
        """A pass over the data of a bucket for its low-rank noise models, one per observation.

        Each tile gives the spectrum and white level of its channels, in its rows of arrays of its
        observation, summed over its tiles on the device that holds them; each observation is
        then fitted there (see [`fit_low_rank`][furax.mapmaking._pass.fit_low_rank]). The detector
        axis of the models is padded to a whole number of tiles.

        Returns the models and their cores, their leaves `(segment, ...)`, on the devices.
        """
        detectors = max(self._buffer_shapes[i].detector_count for i in bucket.observations)
        capacity = -(-detectors // bucket.tile_size) * bucket.tile_size
        sample_rate = self._probed_sample_rate
        reader.reset_failures()  # fresh pass: drop failures recorded by any previous read
        reads = tile_reads(bucket, reader, self.placement.axis, self._prefetches_reads)
        reads = reads.zip(_first_detectors(bucket))
        kernel = fit_kernel(self.fit_config, self.landscape, capacity, sample_rate)
        (spectra, level, n_real), _ = map_reduce(
            kernel, reads, ('segment_sum', 'segment_sum', 'segment'), sky
        )
        models, cores = fit_low_rank(
            spectra,
            level,
            n_real,
            self.fit_config,
            self.landscape,
            capacity,
            bucket.envelope.sample_count,
            bin_edges(self.fit_config.weighting.low_rank, sample_rate, self._shortest(bucket)),
            sample_rate,
            self.placement.axis,
        )
        return models, cores

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
            models.append(GapSplines(spacing, budget, *self._bridge(bucket)))
        return models

    def _bridge(self, bucket: TileBucket) -> tuple[int, float, float]:
        """The amplitude slots of the bridge of a bucket, its knot spacing in samples, its growth.

        The slots fit the longest padding of the observations of the bucket; there are none
        without a bridge.
        """
        if not self._bridge_samples:
            return 0, 1.0, 1.0
        bridge = self.config.weighting.low_rank.bridge
        assert bridge is not None
        spacing = bridge.resolution * self._probed_sample_rate
        envelope = bucket.envelope.sample_count
        shortest = self._shortest(bucket)
        slots = bridge_amplitudes(envelope - shortest, spacing, bridge.growth)
        logger.info(
            f'Bridge: {slots} amplitudes per channel over at most {envelope - shortest} samples'
        )
        return slots, spacing, bridge.growth

    def _shortest(self, bucket: TileBucket) -> int:
        """The samples of the shortest observation of a bucket, before their padding."""
        return min(self._probe_shapes[i].sample_count for i in bucket.observations)

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
        return max(math.ceil(bridge.length * self._probed_sample_rate), 4)

    @cached_property
    def _probed_sample_rate(self) -> float:
        """The sample rate shared by the observations, from their probes.

        The low-rank noise model places its frequency bins at it, which must be known before
        compilation. An observation whose probe failed is left out here; its read fails later as
        well.
        """
        rates = np.array(
            [
                p.sample_rate
                for p in self._catalog.probes.values()
                if isinstance(p, ProbeRecord) and p.sample_rate is not None
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
        # the low-rank noise model places its bins at the sample rate
        rate = explicit or self.config.weighting.mode == WeightingMode.LOW_RANK
        # the good ranges size the gap amplitudes (`ExplicitGapsConfig.probe`)
        count_gaps = explicit and self.config.gaps.explicit.probe
        local: dict[int, ProbeRecord | str] = {}
        for i in range(jax.process_index(), self.n_observations, n_proc):
            try:
                probe = self.observations[i].probe(need_intervals, rate, count_gaps)
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
        return ProbeRecord(probe.shape, spacing, count, probe.sample_rate)

    @property
    def _explicit_gaps(self) -> bool:
        """Whether the gaps are explicit, their amplitudes solved for with the map."""
        return self.config.gap_treatment == GapTreatment.EXPLICIT


def _first_detectors(bucket: TileBucket) -> Array:
    """The first detector of each tile of a bucket in its observation, `(tile,)`."""
    return jnp.asarray(bucket.tile_starts, jnp.int32)


def _marginalized_modes(
    tiles: StackedEnsemble, out: dict[str, Any], diagnostics: DiagnosticsConfig
) -> dict[str, Any]:
    """The statistics of a bucket under a low-rank noise model, split between its modes and the rest.

    `out` holds those under the diagonal weight, and the sums `modes` and the low-frequency
    coefficients `low` of the residual that give their correction (see
    [`statistics_correction`][furax.mapmaking._diagnostics.statistics_correction]). The
    statistics of the modes, `modes_chi2` and `modes_samples`, are per segment (see
    [`modes_statistics`][furax.mapmaking._diagnostics.modes_statistics]).
    """
    z = modes_core(tiles)(out['modes'])
    zipped = tiles.zip((out['low'], z), strides=(1, 'segment'))

    def correct(member):
        tile, (low, z_o) = member
        return statistics_correction(tile, low, z_o, diagnostics)

    names = [n for n in ('chi2', 'bands') if getattr(diagnostics, n)]
    correction, _ = map_reduce(correct, zipped, dict.fromkeys(names, 1))
    # the mode covariance of each observation, which its tiles share
    cores = lift(lambda tile: tile.noise_model.core, tiles)
    template = tiles.template
    modes = modes_statistics(
        z, cores.by_segment().stacked, template.noise_model, template.tod_format.n_samples
    )
    return out | modes | {name: out[name] - correction[name] for name in names}


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
