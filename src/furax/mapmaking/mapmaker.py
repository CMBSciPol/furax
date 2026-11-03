import logging
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
from ._bucket import Batch, BucketModel
from ._catalog import ObservationCatalog, Read, detector_table, gather_rows
from ._injection import inject, load_sky, unpadded
from ._landscape import build_landscape
from ._model import (
    FACTORED_KINDS,
    ObservationModel,
    ObservationTemplates,
    OutKind,
    acquisition_operator,
    restrict_streams,
)
from ._observation import AbstractLazyObservation, ObservationBufferShape, ReaderField
from ._provenance import provenance
from ._reader import ObservationReader, slice_detector_axis, slice_detectors
from .config import GapTreatment, InjectionMode, MapMakingConfig, NoiseSource, WeightingMode
from .gap_filling import gap_fill
from .layout import Bucket, SlotLayout, SlotPlacement
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
        local: dict[int, ObservationBufferShape | str] = {}
        for i in range(jax.process_index(), self.n_observations, n_proc):
            try:
                local[i] = self.observations[i].probe_shape(need_intervals)
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
        """The run configuration, with the noise fit options completed from the probes.

        [`padding_aware_welch`][furax.mapmaking.noise.padding_aware_welch] drops the segments that
        fall in an observation's padded tail, but the segment length sets the number of frequency
        bins and so cannot vary with the padding. A segment longer than the shortest observation
        would therefore leave one partial segment straddling that observation's padding, and its
        noise fit would see the zeros. Clipping the segment keeps every averaged segment inside
        real data. The segments of the low-rank fit are clipped likewise.

        The clip is taken over the whole dataset rather than over each bucket, so that the noise
        fit stays independent of how the observations happen to be bucketed. Observation lengths
        come from the probe, which is an upper bound: an observation that reads shorter than probed
        still falls back to a partial segment.
        """
        weighting = self.config.weighting
        if weighting.mode == WeightingMode.IDENTITY or weighting.source != NoiseSource.FIT:
            return self.config  # no PSD is estimated, so the segment length is unused
        shortest = min(shape.sample_count for shape in self._probe_shapes)
        if weighting.mode == WeightingMode.LOW_RANK:
            low_rank = replace(
                weighting.low_rank, nperseg=min(weighting.low_rank.nperseg, shortest)
            )
            return replace(self.config, weighting=replace(weighting, low_rank=low_rank))
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

        The probes are dealt round-robin to the processes and gathered, as for the shapes. An
        observation whose probe fails is left out here; its read fails later as well.
        """
        n_proc = jax.process_count()
        local: dict[int, float] = {}
        for i in range(jax.process_index(), self.n_observations, n_proc):
            try:
                local[i] = float(self.observations[i].probe_sample_rate())
            except Exception:
                logger.exception('Sample rate probe of observation %d failed', i)
        rates = np.array(list(gather_rows(local).values()))
        if rates.size == 0:
            raise RuntimeError('no observation gave its sample rate')
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
        # Every slot of a bucket is padded to its reader's out_structure, real or empty.
        real_bytes = sum(reader.total_nbytes for reader in self.readers)
        padded_bytes = 0
        for b, (bucket, reader) in enumerate(zip(layout.buckets, self.readers, strict=True)):
            slot_bytes = furax.tree.nbytes(reader.out_structure)
            padded_bytes += slot_bytes * bucket.n_slots
            logger.info(
                f'Bucket {b}: {bucket.summary(layout.n_devices)} '
                f'slot_size={format_bytes(slot_bytes)} '
                f'real_size={format_bytes(reader.total_nbytes)} '
                f'pad_size={format_bytes(slot_bytes * bucket.n_slots - reader.total_nbytes)}'
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
            noise_parameters = None
            if self.config.weighting.mode != WeightingMode.IDENTITY:
                noise_parameters = acc.noise_parameters(self.layout, self.placement)

            BJ = acc.block_jacobi(self.placement, self.config.binned, self.landscape.structure)
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
            system = acc.build_system(S, BJ, self.layout, self.placement, response_sky)
            if (two_level := self.config.solver.two_level) is not None:
                system = system.deflate(two_level)
            solved = system.solve(
                self.config.solver,
                self.placement,
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
            policy=Policy(axis=self.placement.axis),
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
        noise_sample_rate = None
        if config.weighting.mode == WeightingMode.LOW_RANK:
            noise_sample_rate = self._probed_sample_rate
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
            # The member is the first detector of its batch: the fields stay whole in their read
            # layout, closed over `apply`, which slices its batch out. Stacking the per-detector
            # fields over a leading batch axis instead would transpose any field with axes ahead
            # of the detector one, such as the demodulated TOD, and XLA materializes the copy.
            starts = batch_size * jnp.arange(n_batches, dtype=jnp.int32)
            ensemble = StackedEnsemble(n_batches, starts)

            def apply(start):
                data = slice_detectors(observation, start, batch_size)
                if injection is None:
                    data = cast_tod(data)
                else:  # the samples to inject into, per detector
                    injectable = slice_detector_axis(injection[1], start, batch_size)
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
                'inverses': 1,
                'grams': {
                    kind: ('sum', None) for kind, held in FACTORED_KINDS.items() if held == 'sum'
                },
                'rhs': {
                    kind: 'sum' if held == 'sum' else 1 for kind, held in FACTORED_KINDS.items()
                },
            }
            out, resolved = map_reduce(apply, ensemble, spec)
            kinds = _out_kinds(out, resolved)
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
            batches, kept = _partition(batch, batch_kinds)
            return out['hits'], out['map'], (batches, kept, out['rhs']['common'])

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

        def accumulate(data, tod, padding, use):
            temperature = None
            if config.read_stokes != landscape.stokes:
                # The extra streams read, the I one for the T2P template, stay out of the model
                temperature = data[ReaderField.SAMPLE_DATA].i
                data, tod = restrict_streams(data, tod, landscape.stokes)
            obs = ObservationModel.create(
                data, padding, config, landscape, noise_sample_rate=noise_sample_rate
            )

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


def _out_kinds(outputs: PyTree[Any], resolved: PyTree[Any]) -> PyTree[OutKind]:
    """The kind of each output leaf of `map_reduce`, in the `OutKind` vocabulary.

    The resolved spec puts `None` on the shared leaves, which `jax.tree.map` over the outputs
    would read as a missing subtree; the returned tree names every kind, so it maps with them.
    """
    treedef = jax.tree.structure(outputs)
    kinds = treedef.flatten_up_to(resolved)
    names = [{1: 'stack', None: 'shared'}.get(kind, kind) for kind in kinds]
    return jax.tree.unflatten(treedef, names)


def _partition(
    outputs: PyTree[Array], kinds: PyTree[OutKind]
) -> tuple[PyTree[Array | None], PyTree[Array | None]]:
    """The `'stack'` outputs, and the others.

    Each is `None` where the other has a leaf; together they are one batch of the observation.
    """
    stacked = jax.tree.map(lambda x, kind: x if kind == 'stack' else None, outputs, kinds)
    others = jax.tree.map(lambda x, kind: None if kind == 'stack' else x, outputs, kinds)
    return stacked, others
