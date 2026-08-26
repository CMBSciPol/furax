import pickle
from abc import abstractmethod
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, fields, replace
from functools import cached_property
from logging import Logger
from math import prod
from pathlib import Path
from typing import Any, ClassVar, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import numpy as np
import pixell.enmap
import pixell.utils
from astropy.io import fits
from astropy.wcs import WCS
from jax import ShapeDtypeStruct
from jax.experimental import multihost_utils as mhu
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jax.typing import ArrayLike
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
    BlockDiagonalOperator,
    IndexOperator,
)
from furax.interfaces.lineax import as_lineax_operator
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

from ._distributed import cross_process_sum, split_stream
from ._geometry import minimum_enclosing_arc
from ._logger import logger as furax_logger
from ._model import ObservationModel, ObservationTemplates
from ._observation import (
    AbstractGroundObservation,
    AbstractLazyObservation,
    FileBackedLazyObservation,
    ObservationBufferShape,
    ReaderField,
)
from ._partition import partition_balanced
from ._reader import ObservationReader
from .config import (
    GapTreatment,
    LandscapeConfig,
    MapMakingConfig,
    Methods,
    NoiseSource,
    WCSConfig,
    WeightingMode,
)
from .gap_filling import gap_fill
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel
from .preconditioner import BJPreconditioner
from .results import MapMakingResults
from .streaming import StreamOperator
from .templates import ATOPProjectionOperator
from .weight import WeightOperator


@register_dataclass
@dataclass(frozen=True)
class AccumulatedModel:
    """Result of [`MultiObservationMapMaker.build_model_and_accumulate`][].

    Attributes:
        model: Per-observation model, sharded over observations.
        templates: Per-observation templates, sharded over observations.
        hit_map: Hit map, replicated (reduced across processes).
        map_rhs: Map RHS, replicated (reduced across processes).
        amplitude_rhs: Explicit-template RHS, obs-stacked.
    """

    model: ObservationModel = field(metadata={'sharded': True})
    templates: ObservationTemplates | None = field(metadata={'sharded': True})
    hit_map: Int64[Array, '...']
    map_rhs: StokesType
    amplitude_rhs: PyTree[Array] | None = field(metadata={'sharded': True})

    @classmethod
    def shard_map_out_specs(cls, axis: str) -> 'AccumulatedModel':
        """Sharding of each field, as an instance of the class, so `jax.shard_map` matches by field.

        Args:
            axis: Name of the mesh axis the observations are distributed over.
        """
        specs = {f.name: P(axis) if f.metadata.get('sharded') else P() for f in fields(cls)}
        return cls(**specs)  # type: ignore[arg-type]


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
        # Build the reader once, up front: it only probes shapes (no mesh context needed)
        rhs_fields = {ReaderField.METADATA, ReaderField.SAMPLE_DATA}
        model_fields = ObservationModel.required_reader_fields(self.config)
        template_fields = ObservationTemplates.required_reader_fields(self.config)
        self.reader_fields = model_fields | rhs_fields | template_fields
        self.reader = self.get_reader(self.reader_fields)

    def _check_config(self) -> None:
        """Validate and adjust config for method-specific compatibility."""
        if self.config.method == Methods.ATOP:
            if not self.config.binned:
                raise ValueError('ATOP requires diagonal weighting (weighting.mode=DIAGONAL).')
            if 'I' in (stokes := self.config.landscape.stokes):
                if stokes != 'IQU':
                    raise ValueError(
                        f'ATOP does not support intensity map reconstruction and {stokes=!r}'
                        " cannot be reduced to a supported type. Use stokes='QU' instead."
                    )
                self.logger.info(
                    "Received stokes='IQU', but ATOP does not support intensity map reconstruction."
                    " Falling back to stokes='QU' instead."
                )
                self.config.landscape.stokes = 'QU'
        if self.config.use_templates and not self.config.binned:
            raise NotImplementedError('Using templates requires diagonal weighting.')
        if self.config.use_templates and self.config.method == Methods.ATOP:
            raise NotImplementedError('ATOP combined with templates is not yet supported.')

    @cached_property
    def mesh(self) -> Mesh:
        # The observation axis stops at the process boundary: each process owns its observations
        # outright, so nothing forces the buffer shapes of one process on any other. A
        # single-process run is the degenerate case, where this is every device in the job.
        #
        # `Auto` rather than `Explicit`: under explicit axis types an array built on this mesh
        # carries it in its sharding and then cannot be closed over inside `shard_map`, which is
        # how the sky landscape reaches the pointing. The price is that every `shard_map` must
        # spell out its `in_specs` instead of having them inferred.
        devices = np.array(sorted(jax.local_devices(), key=lambda d: d.id))
        return Mesh(devices, ('obs',), axis_types=(AxisType.Auto,))

    @property
    def sharding(self) -> NamedSharding:
        return NamedSharding(self.mesh, P('obs'))

    @overload
    def distribute(self, x: ArrayLike) -> Array: ...
    @overload
    def distribute[S](self, x: S) -> S: ...
    def distribute(self, x: Any) -> Any:
        """Shard a pytree of process-local arrays along the leading 'obs' axis."""
        return jax.tree.map(lambda a: jax.device_put(a, self.sharding), x)

    @property
    def n_observations(self) -> int:
        """Total number of observations across all processes."""
        return len(self.observations)

    @cached_property
    def _probe(self) -> tuple[list[ObservationBufferShape], np.ndarray]:
        """Every observation's probe shape, plus a mask of the ones that could not be probed.

        Assignment needs the shapes of *all* observations, but probing one costs an open, so they
        are split evenly for this pass only -- each process probes a share and the all-gather
        inside ``_gather_shapes`` gives everyone the full list. The real assignment is decided
        from it in [`owned_observations`][], and the shapes are handed to the reader so the probe
        is not paid twice.
        """
        start, n_owned, _ = get_obs_distribution_to_process(self.n_observations)
        provisional = tuple(range(start, start + n_owned))
        shapes, failed = ObservationReader._gather_shapes(
            self.observations, provisional, self.reader_fields
        )
        local = np.zeros(self.n_observations, dtype=bool)
        local[list(failed)] = True
        gathered = np.asarray(mhu.process_allgather(local)).reshape(-1, self.n_observations)
        return shapes, gathered.any(axis=0)

    @cached_property
    def owned_observations(self) -> np.ndarray:
        """Global indices of the observations this process owns.

        The observations are partitioned so that every process gets a comparable amount of work
        and pads its buffers to a shape close to its own observations. A process holding short
        scans therefore holds more of them.
        """
        shapes, _ = self._probe
        groups = partition_balanced(shapes, jax.process_count())
        return np.asarray(groups[jax.process_index()], dtype=int)

    @property
    def obs_distribution(self) -> tuple[int, int]:
        """``(n_owned, n_pad)`` for this process."""
        n_owned = len(self.owned_observations)
        n_pad = -n_owned % jax.local_device_count()
        return n_owned, n_pad

    def get_padded_read_indices(self) -> np.ndarray:
        _, n_pad = self.obs_distribution
        return np.pad(self.owned_observations, (0, n_pad), mode='edge')

    def get_reader(self, required_fields: Collection[str]) -> ObservationReader[T]:
        """Build an ObservationReader for this process's local observations."""
        # Pass padded indices so the reader sizes its buffers from the slots actually read; the
        # count differs between ranks, since each pads only to fill its own devices.
        shapes, failed = self._probe
        return ObservationReader.from_observations(
            self.observations,
            read_indices=tuple(self.get_padded_read_indices()),
            requested_fields=required_fields,
            demodulated=self.config.demodulated,
            stokes=self.config.landscape.stokes,
            dtype=self.config.dtype,
            shapes=shapes,
            known_failures=[int(i) for i in np.flatnonzero(failed)],
        )

    def run(self, out_dir: str | Path | None = None) -> MapMakingResults:
        """Runs the mapmaker and return results after saving them to the given directory."""
        results = self.make_maps()

        # Save outputs on process 0 only (all processes hold the same replicated result)
        if out_dir is not None and jax.process_index() == 0:
            out_dir = Path(out_dir).resolve()
            results.save(out_dir)
            self.logger.info(f'saved results to {out_dir}')
            self.config.dump_yaml(out_dir / 'mapmaking_config.yaml')
            self.logger.info('saved mapmaking configuration to file')

        # Barrier so other ranks don't race ahead while rank 0 is still writing.
        mhu.sync_global_devices('mapmaker.run.save_done')

        return results

    def make_maps(self) -> MapMakingResults:
        """Computes the mapmaker results (maps and other products)."""
        logger_info = lambda msg: self.logger.info(f'MultiObsMapMaker: {msg}')

        n_processes = jax.process_count()
        rank = jax.process_index()
        n_local_devices = jax.local_device_count()
        n_devices = jax.device_count()

        # Information about how observations are distributed among processes
        n_owned, n_pad = self.obs_distribution
        n_per_proc = n_owned + n_pad
        n_per_dev = n_per_proc // n_local_devices
        # Slot counts and buffer sizes now differ between processes, so the totals have to be
        # summed rather than scaled from this process's share.
        n_slots_global = int(np.asarray(mhu.process_allgather(np.array([n_per_proc]))).sum())

        # Every slot (real or padding) is padded to the same reader.out_structure -- the same
        # only within this process, which sizes its buffers from its own observations.
        per_slot_bytes = furax.tree.nbytes(self.reader.out_structure)
        local_bytes = per_slot_bytes * n_per_proc
        global_bytes = int(np.asarray(mhu.process_allgather(np.array([local_bytes]))).sum())

        # The true, total data size (before observations are padded to a common structure)
        real_bytes = self.reader.total_nbytes

        slot_overhead = (n_slots_global - self.n_observations) / self.n_observations
        byte_overhead = (global_bytes - real_bytes) / real_bytes
        logger_info(
            f'layout procs={n_processes} dev_per_proc={n_local_devices} dev_total={n_devices}'
        )
        logger_info(
            f'dataset obs={self.n_observations} slots={n_slots_global} slot_overhead=+{slot_overhead:.1%} '
            f'slots_per_proc={n_per_proc} slots_per_dev={n_per_dev} slot_size={format_bytes(per_slot_bytes)}'
        )
        logger_info(
            f'dataset real={format_bytes(real_bytes)} global={format_bytes(global_bytes)} '
            f'byte_overhead=+{byte_overhead:.1%}'
        )

        rank_pad = n_pad / n_per_proc
        # Segments are volume-balanced, so the counts differ between ranks; the sample range says
        # how homogeneous this rank's share is, which is what its buffer padding costs it.
        owned_samples = [self._probe[0][i].sample_count for i in self.owned_observations]
        logger_info(
            f'rank={rank} real={n_owned} pad={n_pad} '
            f'samples={min(owned_samples)}..{max(owned_samples)} '
            f'pad_pct={rank_pad:.1%} real_size={format_bytes(per_slot_bytes * n_owned)} '
            f'pad_size={format_bytes(per_slot_bytes * n_pad)}'
        )

        # The mesh covers this process's devices only, so every stream-axis reduction inside it
        # needs finishing across processes.
        with jax.set_mesh(self.mesh), split_stream():
            # Single read pass, so each observation is read/preprocessed exactly once.
            acc = self.build_model_and_accumulate()
            jax.block_until_ready(acc)
            logger_info('Accumulated hit map and RHS vector')

            failed_observations = self._collect_failed_observations()
            if failed_observations:
                logger_info(f'{len(failed_observations)} observation(s) failed and were excluded')

            # Diagonal pixel system for the block-Jacobi preconditioner
            H_sky: AbstractLinearOperator = StreamOperator.column(acc.model.H)
            W: AbstractLinearOperator = StreamOperator.diagonal(acc.model.W)
            W_diag = (
                W
                if self.config.binned
                else StreamOperator.diagonal(eqx.filter_vmap(ObservationModel.diag_W)(acc.model))
            )
            # Specify leading axis dimension because F can be trivial (no array leaves)
            # Must be this process's slot count to compose with the other operators, whose own
            # leaves are sharded over the local devices and report exactly that.
            F = StreamOperator.diagonal(acc.model.F, n_lead=n_per_proc)
            A_diag = (H_sky.T @ W_diag @ F @ H_sky).reduce()
            BJ = BJPreconditioner.create(A_diag)
            icov = BJ.blocks.block_until_ready()
            logger_info('Computed white noise inverse covariance')

            # Fold ATOP deprojector into the weight from now on
            W = (W @ F).reduce()

            # Pixel selection from the icov estimate
            hit_map = acc.hit_map  # rebound below, once the pixel selection is known
            valid_pixels = self.pixel_selection(hit_map, icov)
            # Select valid pixels on the (trailing) sky axes, leaving the leading Stokes axis intact.
            S = IndexOperator((..., *jnp.where(valid_pixels)), in_structure=H_sky.in_structure)
            H_sky_selected = H_sky @ S.T  # takes the selected pixels; H_sky takes the full grid
            M_sky = (S @ BJ.I @ S.T).reduce()

            n_selected = jnp.sum(valid_pixels)
            n_observed = jnp.sum(hit_map > 0)
            n_total = valid_pixels.size
            logger_info(f'Selected {n_selected} pixels ({n_observed} seen, {n_total} total)')

            hit_map = hit_map.at[~valid_pixels].set(0)  # excluded pixels have zero hits
            icov = jnp.moveaxis(icov, [-2, -1], [0, 1])  # (*pixels, ns, ns) → (ns, ns, *pixels)

            # Unified GLS solve (Hᵀ W' H) x = Hᵀ W' d  (W already bundles the sample mask).
            #
            # H maps the unknowns to TOD:
            # - no templates / implicit only: H = H_sky        (sky map only);
            # - explicit templates:          H = [H_sky | Tₑ] (sky map + template amplitudes).
            #
            # Implicit templates fold into the weight (W → W', deprojection).

            templates = acc.templates
            # Fold implicit templates into the system weight (marginal deprojection W').
            if templates is not None and (implicit := templates.implicit) is not None:
                Ti = StreamOperator.diagonal(implicit.operator)
                G = StreamOperator.diagonal(implicit.gram_inverse)
                W = (W - W @ Ti @ G @ Ti.T @ W).reduce()

            explicit = templates.explicit if templates is not None else None
            if explicit is None:
                M = M_sky
                rhs_joint: Any = S(acc.map_rhs)
                A = (H_sky_selected.T @ W @ H_sky_selected).reduce()
            else:
                Te = StreamOperator.diagonal(explicit.operator)
                G_e = StreamOperator.diagonal(explicit.gram_inverse)
                M = BlockDiagonalOperator([M_sky, G_e])
                rhs_joint = [S(acc.map_rhs), acc.amplitude_rhs]

                # Joint sky + explicit-amplitude system
                H_joint = StreamOperator.block_row([H_sky, Te])
                A_joint = (H_joint.T @ W @ H_joint).reduce()
                select = BlockDiagonalOperator([S, IdentityOperator(in_structure=Te.in_structure)])
                A = (select @ A_joint @ select.T).reduce()

            iteration_callback = None
            if self.config.solver.verbose:
                # log from rank 0 only
                def iteration_callback(step: Array, r_norm: Array) -> None:
                    if rank == 0:
                        logger_info(f'CG step={int(step)} residual={float(r_norm):.6e}')

            result = furax.linalg.cg(
                A,
                rhs_joint,
                preconditioner=M,
                iteration_callback=iteration_callback,
                **self.config.solver.options,
            )
            logger_info(f'Finished GLS solve ({int(result.num_steps)} it)')

            if explicit is not None:
                sky_estimate, amplitudes = result.solution
            else:
                sky_estimate = result.solution
                amplitudes = None

        if amplitudes is not None:
            # Gather the (sharded) amplitudes on rank 0. Result is host-side.
            # Gathering the slot mask the same way drops the padding slots.
            real = mhu.process_allgather(self._real_observation_mask(), tiled=True)
            gathered = mhu.process_allgather(amplitudes, tiled=True)
            amplitudes = jax.tree.map(lambda a: a[real], gathered)

        return MapMakingResults(
            map=S.T(sky_estimate),  # all sky pixels including those not estimated (zero)
            icov=icov,
            hit_map=hit_map,
            solver_stats={'num_steps': int(result.num_steps)},
            landscape=self.landscape,
            failed_observations=failed_observations,
            template_amplitudes=dict(amplitudes) if amplitudes is not None else None,
        )

    def _collect_failed_observations(self) -> list[str]:
        """Names of observations that failed to load, gathered across all processes.

        Each process records the local indices it could not read (``reader.failed_indices``); a
        boolean mask over all observations is all-gathered and OR-reduced so every process reports
        the same global set.
        """
        local = np.zeros(self.n_observations, dtype=bool)
        local[sorted(self.reader.failed_indices)] = True
        gathered = np.asarray(mhu.process_allgather(local)).reshape(-1, self.n_observations)
        failed = np.flatnonzero(gathered.any(axis=0))
        return [self.observations[int(i)].name for i in failed]

    def build_model_and_accumulate(self) -> AccumulatedModel:
        """Build the model and accumulate the hit map and map RHS in one pass over the data.

        Padding (or failed) slots are gated out using a zero-mask, so they drop from the system.

        This method must run under ``jax.set_mesh(self.mesh)``.
        """
        config = self.config
        landscape = self.landscape
        reader = self.reader
        reader.reset_failures()  # fresh pass: drop failures recorded by any previous read
        fill_gaps = config.gaps.treatment == GapTreatment.FILL and not config.binned
        build_templates = config.use_templates

        indices = self.distribute(self.get_padded_read_indices())
        is_real = self.distribute(self._real_observation_mask())
        axis = jax.sharding.get_abstract_mesh().axis_names[0]

        def kernel(indices, is_real):  # type: ignore[no-untyped-def]
            def step(carry, args):  # type: ignore[no-untyped-def]
                hits_acc, rhs_acc = carry
                i, real = args

                # Skip the load for padding slots: only the real branch hits the io_callback,
                # so a padded observation is never read or preprocessed just to be masked away.
                data, padding, valid = jax.lax.cond(
                    real,
                    lambda: reader.read(i),
                    lambda: reader.read_filler(),
                )
                obs = ObservationModel.create(data, padding, config, landscape)

                # Padding/failed observations contribute nothing
                obs.M = obs.M.restrict(real & valid)

                # Hit map = nearest-neighbour coverage of the sample mask
                hit_pointing = PointingOperator.create(
                    landscape,
                    data[ReaderField.BORESIGHT_QUATERNIONS],
                    data[ReaderField.DETECTOR_QUATERNIONS],
                ).as_stokes_i(interpolate=False)
                # Read the mask directly: M(ones) = M.to_boolean_mask()
                masked_tod = obs.M.to_boolean_mask()
                # The mask is (ndet, nsamp) even in the demodulated case (all legs share the same)
                masked = masked_tod.data if isinstance(masked_tod, Stokes) else masked_tod
                hits_i = jnp.int64(hit_pointing.T(StokesI(masked)).i)

                # RHS contribution (optionally gap-filled).
                def func_gapfill(tod):  # type: ignore[no-untyped-def]
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
                        tod,
                        obs.W.weight,
                        obs.M,
                        rate=obs.sample_rate,
                        max_cg_steps=config.gaps.fill_options.max_steps,
                        rtol=config.gaps.fill_options.rtol,
                        preconditioner=preconditioner,
                        metadata=data[ReaderField.METADATA],
                    )

                # Use Python `if` for static conditions, so inactive branches are not traced.
                tod = data[ReaderField.SAMPLE_DATA]
                if fill_gaps:
                    tod = jax.lax.cond(
                        real & valid,
                        func_gapfill,
                        lambda _: _,  # return raw data as-is
                        tod,
                    )

                if not build_templates:
                    if fill_gaps:
                        # Gaps filled: skip the data-side mask so the fill survives N⁻¹.
                        rhs_i = obs.rhs_operator_prefilled(tod)
                    else:
                        rhs_i = obs.rhs_operator(tod)
                    carry = (hits_acc + hits_i, furax.tree.add(rhs_acc, rhs_i))
                    return carry, (obs, None, None)

                # Templates require config.binned=True, so fill_gaps never applies here.
                templates, wd = ObservationTemplates.create(data, config, obs, tod)
                explicit = templates.explicit
                rhs_i = obs.H.T(wd)
                amp_i = explicit.operator.T(wd) if explicit is not None else None
                carry = (hits_acc + hits_i, furax.tree.add(rhs_acc, rhs_i))
                return carry, (obs, templates, amp_i)

            init_hits = jax.lax.pcast(jnp.zeros(landscape.shape, jnp.int64), axis, to='varying')
            init_rhs = jax.lax.pcast(landscape.zeros(), axis, to='varying')
            (hits, rhs), stacked = jax.lax.scan(step, (init_hits, init_rhs), (indices, is_real))
            hits, rhs = jax.lax.psum((hits, rhs), axis)
            model, templates, amp_rhs = stacked
            return AccumulatedModel(
                model=model,
                templates=templates,
                hit_map=hits,
                map_rhs=rhs,
                amplitude_rhs=amp_rhs,
            )

        # Both inputs are distributed over the observation axis; the mesh's axis types are `Auto`,
        # so the specs are given rather than inferred.
        in_specs = (P('obs'), P('obs'))
        out_specs = AccumulatedModel.shard_map_out_specs(axis)
        accumulated = jax.shard_map(in_specs=in_specs, out_specs=out_specs, check_vma=False)(
            kernel
        )(indices, is_real)
        # The psum inside the kernel spans this process's devices only, so these are still partials.
        hits, rhs = cross_process_sum((accumulated.hit_map, accumulated.map_rhs))
        return replace(accumulated, hit_map=hits, map_rhs=rhs)  # type: ignore[no-any-return]

    def _real_observation_mask(self) -> np.ndarray:
        """Boolean flag per padded slot: True for real observations, False for padding."""
        n_owned, n_pad = self.obs_distribution
        return np.concatenate([np.ones(n_owned, dtype=bool), np.zeros(n_pad, dtype=bool)])

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


def get_obs_distribution_to_process(
    n_obs: int,
    rank: int | None = None,
    n_proc: int | None = None,
    n_local: int | None = None,
) -> tuple[int, int, int]:
    """Compute this process's slice for distributed mapmaking.

    Distributes ``n_obs`` observations across processes as evenly as possible
    (first ``n_obs % n_proc`` processes get one extra), then pads each process's
    share to the next multiple of ``n_local`` so every local device has a uniform
    workload.  Processes need not end up with the same number of slots: each owns
    its observations outright, so only its own devices have to divide evenly.

    Args:
        n_obs: Total number of observations across all processes.
        rank: Process index. Defaults to ``jax.process_index()``.
        n_proc: Process count. Defaults to ``jax.process_count()``.
        n_local: Local device count. Defaults to ``jax.local_device_count()``.

    Returns:
        A tuple ``(start, n_owned, n_pad)`` where ``start`` is the index of the
        first real observation owned by this process, ``n_owned`` is the number
        of real observations, and ``n_pad`` is the number of padding slots so that
        ``n_owned + n_pad`` is a multiple of ``n_local``.

    Raises:
        ValueError: If ``n_obs < n_proc``.
    """
    if rank is None:
        rank = jax.process_index()
    if n_proc is None:
        n_proc = jax.process_count()
    if n_local is None:
        n_local = jax.local_device_count()

    if n_obs < n_proc:
        raise ValueError(
            f'Not enough observations ({n_obs}) for {n_proc} processes. '
            f'Provide more observations or run with fewer processes.'
        )

    base = n_obs // n_proc
    remainder = n_obs % n_proc
    n_owned = base + (1 if rank < remainder else 0)
    start = rank * base + min(rank, remainder)

    # Nothing has to agree between processes, so pad only enough to fill this process's own
    # devices; a process owning fewer observations carries fewer empty slots.
    n_per_proc = n_owned + (-n_owned) % n_local
    n_pad = n_per_proc - n_owned

    return start, n_owned, n_pad


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
        assert wcs_config.patch is not None  # mypy: has_geometry guarantees patch is set here
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
            Methods.ATOP: ATOPMapMaker,
        }[config.method]

        if logger is None:
            return maker(config)  # type: ignore[abstract]
        else:
            return maker(config, logger=logger)  # type: ignore[abstract]

    @classmethod
    def from_yaml(cls, path: str | Path, logger: Logger | None = None) -> 'MapMaker':
        return cls.from_config(MapMakingConfig.load_yaml(path), logger=logger)

    def get_landscape(self, observation: AbstractGroundObservation[Any]) -> StokesLandscape:
        """Landscape used for mapmaking with given observation."""
        lc = self.config.landscape
        if (landscape := _static_landscape(lc, self.config.dtype)) is not None:
            return landscape
        assert lc.wcs is not None  # mypy: _static_landscape returns None only for auto WCS
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
                jnp.asarray(observation.get_boresight_quaternions()),
                jnp.asarray(observation.get_detector_quaternions()),
                batch_size=self.config.pointing.batch_size,
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
            meta = {
                'shape': (observation.n_detectors, observation.n_samples),
                'stokes': landscape.stokes,
                'dtype': self.config.dtype,
            }
            polarizer = LinearPolarizerOperator.create(
                **meta,  # type: ignore[arg-type]
                angles=jnp.asarray(
                    observation.get_detector_offset_angles().astype(self.config.dtype)[:, None]
                ),
            )
            hwp = HWPOperator.create(
                **meta,  # type: ignore[arg-type]
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
        def process(d):  # type: ignore[no-untyped-def]
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
            output['noise_fit'] = noise_model.to_array()  # type: ignore[assignment]
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
                StokesIQU.from_iquv(
                    i=jnp.ones(selector.out_structure.shape, dtype=data.dtype),
                    q=jnp.zeros(selector.out_structure.shape, dtype=data.dtype),
                    u=jnp.zeros(selector.out_structure.shape, dtype=data.dtype),
                    v=None,  # type: ignore[arg-type]
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
        def process(d):  # type: ignore[no-untyped-def]
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


class ATOPMapMaker(MapMaker):
    """Class for ATOP mapmaking with diagonal noise covariance."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validation on config
        if not self.config.binned:
            raise ValueError('ATOP Mapmaker is currently incompatible with binned=False')
        if self.config.atop_tau < 2:
            raise ValueError('ATOP tau should be at least 2')
        if self.config.landscape.stokes != 'QU':
            raise ValueError('ATOP only compatible with stokes=QU')

    def make_map(self, observation: AbstractGroundObservation[Any]) -> dict[str, Any]:
        config = self.config
        logger_info = lambda msg: self.logger.info(f'ATOP Mapmaker: {msg}')

        # Data and landscape
        data = jnp.asarray(observation.get_tods(), dtype=config.dtype)
        data_struct = ShapeDtypeStruct(data.shape, data.dtype)
        landscape = self.get_landscape(observation)

        # Acquisition (I, Q, U Maps -> TOD)
        acquisition = self.get_acquisition(observation, landscape=landscape)
        logger_info('Created acquisition operator')

        # ATOP projector
        atop_projector = ATOPProjectionOperator(self.config.atop_tau, in_structure=data_struct)

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
        tau_mask = jnp.abs(atop_projector(masker(jnp.ones_like(data)))) < 0.5 / config.atop_tau
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
        ap = inv_noise @ atop_projector
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


class IQUModulationOperator(AbstractLinearOperator):
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


class QUModulationOperator(AbstractLinearOperator):
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
