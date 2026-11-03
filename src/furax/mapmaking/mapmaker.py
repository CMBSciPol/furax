from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import cached_property, partial
from logging import Logger
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pixell.enmap
import pixell.utils
from fastquat import Quaternion
from jax.experimental import multihost_utils as mhu
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, DTypeLike, Float, Int64, Integer, PyTree

import furax.linalg
import furax.tree
from furax.core import IndexOperator
from furax.obs.landscapes import HealpixLandscape, StokesLandscape, WCSLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, StokesI, StokesType, ValidStokesLiteral
from furax.profiling import format_bytes

from ._accumulated import AccumulatedModel
from ._bucket import Batch, BucketModel
from ._catalog import ObservationCatalog, Read, detector_table, gather_rows
from ._engine import DetectorFamily, Family, StackedFamily, ZippedFamily, family_apply, partition
from ._geometry import minimum_enclosing_arc
from ._injection import inject, load_sky, unpadded
from ._logger import logger as furax_logger
from ._model import (
    FACTORED_KINDS,
    ObservationModel,
    ObservationTemplates,
    acquisition_operator,
    restrict_streams,
)
from ._observation import (
    AbstractLazyObservation,
    FileBackedLazyObservation,
    ObservationBufferShape,
    ReaderField,
)
from ._provenance import provenance
from ._reader import ObservationReader
from .config import (
    GapTreatment,
    InjectionMode,
    LandscapeConfig,
    MapMakingConfig,
    NoiseSource,
    WCSConfig,
    WeightingMode,
)
from .gap_filling import gap_fill
from .layout import Bucket, SlotLayout, SlotPlacement
from .results import CONFIG_FILE, PROVENANCE_FILE, MapMakingResults, write_json
from .templates.gram import factor
from .weight import WeightOperator


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
        self.landscape = (
            _static_landscape(self.config.landscape, self.config.dtype)
            or self._scan_wcs_footprint()
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
        minimum = self.config.pomme_tau if self.config.pomme_tau is not None else 1
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
            self.logger.info(f'saved mapmaking configuration and provenance to {out_dir}')

        # Loaded now, once the run is on record, so that a bad injection fails before the passes
        # over the data, and shows in the output directory
        _ = self.sky
        results = self.make_maps(checkpoint_dir=out_dir, resume=resume)

        # Save outputs on process 0 only (all processes hold the same maps)
        if is_writer:
            results.save(out_dir)
            self.logger.info(f'saved results to {out_dir}')

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
            acc = self.build_model_and_accumulate()
            jax.block_until_ready(acc)
            self._info('Accumulated hit map and RHS vector')

            self._catalog.reads = self._gather_reads()
            if n_failed := self._catalog.n_failed:
                self._info(f'{n_failed} observation(s) failed and were excluded')
            noise_parameters = None
            if self.config.weighting.mode != WeightingMode.IDENTITY:
                noise_parameters = acc.noise_parameters(self.layout, self.placement)

            BJ = acc.block_jacobi(self.placement, self.config.binned, self.landscape.structure)
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
            system = acc.build_system(S, BJ, self.layout, self.placement, response_sky)
            if (two_level := self.config.solver.two_level) is not None:
                system = system.deflate(two_level, self.logger)
            solved = system.solve(
                self.config.solver,
                self.placement,
                self.logger,
                None if checkpoint_dir is None else Path(checkpoint_dir),
                resume,
            )
            sky, amplitudes = solved.x, None
            if system.has_amplitudes:
                sky, *blocks = solved.x
                amplitudes = acc.template_amplitudes(blocks, self.layout, self.placement)

        # Built only now, so that their copies are not held through the solve
        return MapMakingResults(
            map=S.T(sky),  # all sky pixels including those not estimated (zero)
            icov=jnp.moveaxis(icov, [-2, -1], [0, 1]),  # (*pixels, ns, ns) → (ns, ns, *pixels)
            hit_map=jnp.where(valid_pixels, acc.hit_map, 0),  # excluded pixels have zero hits
            solver_stats=solved.solver_stats,
            landscape=self.landscape,
            template_amplitudes=amplitudes,
            injected_map=None if solved.injected_sky is None else S.T(solved.injected_sky),
            observations=self._catalog.to_table(),
            # the noise model covers the streams of the model's TOD, not all those read
            detectors=detector_table(
                self._catalog.reads,
                noise_parameters,
                acc.buckets[0].batches.model.tod_format.stokes,
            ),
        )

    def _info(self, message: str) -> None:
        self.logger.info(f'MultiObsMapMaker: {message}')

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

    def build_model_and_accumulate(self) -> AccumulatedModel:
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
        return AccumulatedModel(buckets=tuple(bucket_models), hit_map=hit_map, map_rhs=map_rhs)

    def _accumulate_bucket(
        self, bucket_index: int, reader: ObservationReader[T], sky: StokesType | None
    ) -> tuple[Int64[Array, '...'], StokesType, BucketModel]:
        """One bucket's pass over the data: its hit map and RHS partials, and its stacked model.

        The observations are accumulated in rounds, each round launching one observation on each
        of this process's devices. A launch frees its working memory (the TOD and what is
        computed from it) when it returns, so only one observation per device is in memory at a
        time, and the processes need not keep in step until the bucket is assembled at the end.
        """
        reader.reset_failures()  # fresh pass: drop failures recorded by any previous read
        bucket = self.layout.buckets[bucket_index]
        device_slots = self.placement.device_slots(bucket_index)  # (local device, round)
        n_rounds = device_slots.shape[1]
        axis = self.placement.axis
        # The launch has no collective, so each process runs it on its own devices without
        # waiting for the others. It still spans the job's mesh: the operators it builds record
        # the mesh they were traced under, and the solve uses them under that mesh.
        launch = jax.jit(
            jax.shard_map(
                self._observation_kernel(bucket),
                in_specs=(P(axis), P(axis), P(axis), P()),
                out_specs=P(axis),
                check_vma=False,
            )
        )

        def load(slot):
            if not bucket.is_empty[slot]:
                return reader.read_host(slot)  # readers are indexed by slot
            # An empty slot is never read or preprocessed just to be masked away
            return reader.read_filler_host()

        hits = rhs = None
        stacked: Any = None
        # The pool reads the observations of a round's devices concurrently
        with ThreadPoolExecutor(len(device_slots)) as pool:

            def read_round(r):
                return list(pool.map(load, device_slots[:, r].tolist()))

            per_device = read_round(0)
            for r in range(n_rounds):
                inputs = self.placement.to_devices(per_device)
                # Hold this round's buffers only as long as its launch: on a CPU device they are
                # its TOD, which the next read must not find still in memory.
                per_device = []
                # dispatched without waiting
                round_hits, round_rhs, out = launch(*inputs, sky)
                del inputs
                hits = round_hits if hits is None else _add(hits, round_hits)
                rhs = round_rhs if rhs is None else _add(rhs, round_rhs)
                stacked = _store_round(stacked, out, r, n_rounds)
                last = r + 1 == n_rounds
                if self._prefetches_reads and not last:
                    per_device = read_round(r + 1)  # while the devices compute
                # wait for the round, so that its working memory is freed
                jax.block_until_ready((hits, rhs, stacked))
                if not self._prefetches_reads and not last:
                    per_device = read_round(r + 1)

        # every device's partial sums, over the job
        hits, rhs = self.placement.replicate((hits, rhs), lambda x: x.sum(axis=0))

        # (device, round, ...) -> (slot, ...), what depends on the detector batch keeping a
        # batch axis after the slot axis
        def merge(tree):
            return self.placement.per_slot(tree, lambda x: x.reshape(-1, *x.shape[2:]), donate=True)

        return hits, rhs, BucketModel(*merge(stacked))

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
    ) -> Callable[[Any, Any, Array, StokesType | None], tuple[Array, StokesType, Any]]:
        """Accumulate one observation per device: its hit map, RHS and model.

        The arguments and results carry a leading axis of size one, the device's, except the sky to
        inject, which every device holds whole.
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

        def kernel(observation, padding, valid, sky):
            observation, padding, use = jax.tree.map(lambda x: x[0], (observation, padding, valid))
            injection = None
            if sky is not None:
                shape = jax.tree.leaves(observation[ReaderField.SAMPLE_DATA])[0].shape
                injection = (sky, unpadded(padding, shape))
            hits, rhs, out = step(observation, padding, use, injection)
            return jax.tree.map(lambda x: x[None], (hits, rhs, out))

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

            # Templates are rejected with gap filling (see `MapMakingConfig`).
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


_add = jax.jit(furax.tree.add, donate_argnums=0)


@partial(jax.jit, donate_argnums=0)
def _set_round(store: PyTree[Array], out: PyTree[Array], r: Array) -> PyTree[Array]:
    return jax.tree.map(lambda a, b: a.at[:, r].set(b), store, out)


def _store_round(
    store: PyTree[Array] | None, out: PyTree[Array], r: int, n_rounds: int
) -> PyTree[Array]:
    """Write a round's outputs, leading with the device axis, as round `r` of the store."""
    if store is None:
        store = jax.tree.map(
            lambda x: jnp.zeros((x.shape[0], n_rounds, *x.shape[1:]), x.dtype, device=x.sharding),
            out,
        )
    return _set_round(store, out, r)
