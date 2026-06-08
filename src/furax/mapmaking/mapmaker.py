import logging
import pickle
import resource
import time
from abc import abstractmethod
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from logging import Logger
from math import prod
from pathlib import Path
from typing import Any, Generic, TypeVar

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
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, DTypeLike, Float, Int64, Integer, PyTree

import furax.linalg
import furax.tree
from furax import (
    AbstractLinearOperator,
    Config,
    DiagonalOperator,
    IdentityOperator,
    MaskOperator,
    OperatorTag,
    SymmetricBandToeplitzOperator,
)
from furax.core import BlockDiagonalOperator, BlockRowOperator, CompositionOperator, IndexOperator
from furax.interfaces.lineax import as_lineax_operator
from furax.obs.landscapes import (
    AstropyWCSLandscape,
    HealpixLandscape,
    StokesLandscape,
    WCSLandscape,
)
from furax.obs.operators import HWPOperator, LinearPolarizerOperator, QURotationOperator
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, StokesI, StokesIQU, StokesPyTreeType, ValidStokesType

from . import templates
from ._geometry import minimum_enclosing_arc
from ._logger import logger as furax_logger
from ._marginalize import build_marginal_weight
from ._model import ObservationModel, ObservationTemplates
from ._observation import AbstractGroundObservation, AbstractLazyObservation
from ._reader import ObservationReader
from ._scan_blocks import (
    ScanBlockColumnOperator,
    ScanBlockDiagonalOperator,
    _obs_axis_size,
    combine_obs_leaves,
    partition_obs_leaves,
)
from .config import (
    GapFillingConfig,
    LandscapeConfig,
    MapMakingConfig,
    Methods,
    NoiseSource,
    WCSConfig,
    WeightingMode,
)
from .gap_filling import GapFillingOperator
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel
from .preconditioner import BJPreconditioner, CGInverseOperator
from .results import MapMakingResults

T = TypeVar('T')


def _log_cg_residuals(
    logger: Logger,
    residuals: Array,
    num_steps: int,
    label: str = 'CG',
) -> None:
    """Log residual progression stored in sol.residuals after a CG solve.

    At DEBUG level logs every iteration; at INFO logs ~5 milestones.
    ``residuals`` has shape ``(max_steps,)`` with zeros past convergence;
    only the first ``num_steps`` entries are meaningful.
    """
    if num_steps == 0:
        return
    arr = np.asarray(residuals[:num_steps])
    if logger.isEnabledFor(logging.DEBUG):
        for i, r in enumerate(arr):
            logger.debug(f'  {label} iter {i}: ||r||={float(r):.4e}')
    else:
        milestones = sorted({0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1})
        for i in milestones:
            logger.info(f'  {label} iter {i}: ||r||={float(arr[i]):.4e}')


class MultiObservationMapMaker(Generic[T]):
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

    def _check_config(self) -> None:
        """Validate and adjust config for method-specific compatibility."""
        if self.config.method == Methods.TWOSTEP:
            if not self.config.binned:
                raise ValueError('TwoStep requires a white noise model (noise.white=True).')
            if not self.config.use_templates:
                raise ValueError('TwoStep requires at least one active template.')
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

    @cached_property
    def mesh(self) -> Mesh:
        return jax.make_mesh((jax.device_count(),), ('obs',))

    @property
    def sharding(self) -> NamedSharding:
        return NamedSharding(self.mesh, P('obs'))

    def distribute(self, x: PyTree) -> PyTree:
        """Shard a pytree of process-local arrays along the leading 'obs' axis."""
        return jax.tree.map(lambda a: jax.make_array_from_process_local_data(self.sharding, a), x)

    @property
    def n_observations(self) -> int:
        """Total number of observations across all processes."""
        return len(self.observations)

    @property
    def obs_distribution(self) -> tuple[int, int, int]:
        """``(start, n_owned, n_pad)`` for this process."""
        return get_obs_distribution_to_process(self.n_observations)

    def get_read_indices(self) -> np.ndarray:
        start, n_owned, _ = self.obs_distribution
        return np.arange(start, start + n_owned)

    def get_padded_read_indices(self) -> np.ndarray:
        _, _, n_pad = self.obs_distribution
        return np.pad(self.get_read_indices(), (0, n_pad), mode='edge')

    def get_reader(self, required_fields: Collection[str]) -> ObservationReader[T]:
        """Build an ObservationReader for this process's local observations."""
        # Pass padded indices: process_allgather inside from_observations needs every
        # rank to send the same shape, so all ranks must report the same obs count.
        return ObservationReader.from_observations(
            self.observations,
            read_indices=tuple(self.get_padded_read_indices()),
            requested_fields=required_fields,
            demodulated=self.config.demodulated,
            stokes=self.config.landscape.stokes,
        )

    def run(self, out_dir: str | Path | None = None) -> MapMakingResults:
        """Runs the mapmaker and return results after saving them to the given directory."""
        t_run = time.perf_counter()
        self.logger.info('Starting mapmaker run')

        results = self.make_maps()
        self.logger.info(f'make_maps completed in {time.perf_counter() - t_run:.1f}s')

        # Save outputs on process 0 only (all processes hold the same replicated result)
        if out_dir is not None and jax.process_index() == 0:
            t_save = time.perf_counter()
            out_dir = Path(out_dir)
            results.save(out_dir)
            self.logger.info(f'saved results to {out_dir} ({time.perf_counter() - t_save:.1f}s)')
            self.config.dump_yaml(out_dir / 'mapmaking_config.yaml')
            self.logger.info('saved mapmaking configuration to file')

        # Barrier so other ranks don't race ahead while rank 0 is still writing.
        mhu.sync_global_devices('mapmaker.run.save_done')

        return results

    def make_maps(self) -> MapMakingResults:
        """Computes the mapmaker results (maps and other products)."""
        if self.config.method == Methods.TWOSTEP:
            return self.make_maps_template()
        n_processes = jax.process_count()
        rank = jax.process_index()
        logger_info = lambda msg: self.logger.info(f'MultiObsMapMaker: {msg}')
        n_local_devices = jax.local_device_count()
        n_devices = jax.device_count()
        start, n_owned, n_pad = self.obs_distribution
        n_per_proc = n_owned + n_pad
        n_per_dev = n_per_proc // n_local_devices
        logger_info(
            f'Layout: {n_processes} process(es) x {n_local_devices} local device(s) = {n_devices} total'
        )
        logger_info(
            f'Observations: {self.n_observations} real, {n_per_proc * n_processes} after padding '
            f'({n_per_proc} per process, {n_per_dev} per device)'
        )
        logger_info(
            f'Rank {rank}: owns obs[{start}:{start + n_owned}] ({n_owned} real + {n_pad} padding)'
        )

        t0 = time.perf_counter()
        t_phase = t0
        model = self.distribute(self.build_model().pad(n_pad))
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logger_info(
            f'Loaded and distributed model in {time.perf_counter() - t_phase:.1f}s'
            f' (RSS {rss_mb:.0f} MB)'
        )
        logger_info('Created system operators')

        with jax.set_mesh(self.mesh):
            t_phase = time.perf_counter()
            hits = accumulate_hits(model)
            jax.block_until_ready(hits)
            logger_info(f'Computed hit map ({time.perf_counter() - t_phase:.1f}s)')

            t_phase = time.perf_counter()
            rhs = self.accumulate_rhs(model)
            jax.block_until_ready(rhs)
            logger_info(f'Accumulated RHS vector ({time.perf_counter() - t_phase:.1f}s)')

            t_phase = time.perf_counter()
            # System operator (full/diagonal)
            A = self.get_system_operator(model)
            diag_A = A if self.config.binned else self.get_system_operator(model, diag=True)
            BJ = BJPreconditioner.create(diag_A)
            icov = BJ.get_blocks().block_until_ready()
            logger_info(
                f'Computed white noise inverse covariance ({time.perf_counter() - t_phase:.1f}s)'
            )

            valid_pixels = self.pixel_selection(hits, icov)
            selector = IndexOperator(jnp.where(valid_pixels), in_structure=A.out_structure)
            n_selected = jnp.sum(valid_pixels)
            n_observed = jnp.sum(hits > 0)
            n_total = valid_pixels.size
            logger_info(f'Selected {n_selected} pixels ({n_observed} seen, {n_total} total)')

            hits = hits.at[~valid_pixels].set(0)  # excluded pixels have zero hits
            icov = jnp.moveaxis(icov, [-2, -1], [0, 1])  # (*pixels, ns, ns) → (ns, ns, *pixels)

            # Solve the mapmaking system
            solver_cfg = self.config.solver
            logger_info(
                f'Starting CG solver (max_steps={solver_cfg.max_steps}, rtol={solver_cfg.rtol})'
            )
            t_phase = time.perf_counter()
            sol = furax.linalg.cg(
                selector @ A @ selector.T,
                selector(rhs),
                preconditioner=selector @ BJ.I @ selector.T,
                **asdict(solver_cfg),
                iteration_callback=(
                    (lambda i, r: self.logger.debug(f'Map CG iter {int(i)}: ||r||={float(r):.4e}'))
                    if self.logger.isEnabledFor(logging.DEBUG)
                    else None
                ),
            )
            estimate = selector.T(sol.solution)
            num_steps = int(sol.num_steps)
            logger_info(
                f'Finished mapmaking CG in {time.perf_counter() - t_phase:.1f}s'
                f' (iterations: {num_steps},'
                f' final ||r||={float(sol.residuals[min(num_steps, solver_cfg.max_steps - 1)]):.4e})'
            )

        logger_info(f'Total make_maps time: {time.perf_counter() - t0:.1f}s')
        return MapMakingResults(
            map=estimate,
            icov=icov,
            hit_map=hits,
            solver_stats={'num_steps': num_steps},
            cg_residuals=sol.residuals,
            landscape=self.landscape,
        )

    def build_model(self) -> ObservationModel:
        """Build the local ObservationModel for this process.

        Each process reads its owned observations. The returned model has
        ``n_owned`` entries along the leading axis (no padding). Padding to the
        uniform per-process count is applied by the caller before sharding.
        """
        reader = self.get_reader(ObservationModel.required_reader_fields(self.config))

        def build_one(_, i):  # type: ignore[no-untyped-def]
            data, padding = reader.read(i)
            return None, ObservationModel.create(data, padding, self.config, self.landscape)

        _, model = jax.lax.scan(build_one, None, self.get_read_indices())
        return model  # type: ignore[no-any-return]

    def build_model_and_templates(self) -> tuple[ObservationModel, ObservationTemplates]:
        if not self.config.use_templates or self.config.templates is None:
            raise RuntimeError('no active templates config')

        # Get a reader for model + templates fields
        model_fields = ObservationModel.required_reader_fields(self.config)
        template_fields = ObservationTemplates.required_reader_fields(self.config)
        reader = self.get_reader(model_fields | template_fields)
        masks_ref = reader.out_structure['valid_sample_masks']
        tod_structure = jax.ShapeDtypeStruct(masks_ref.shape, self.config.dtype)

        def build_one(_, i):  # type: ignore[no-untyped-def]
            data, padding = reader.read(i)
            model = ObservationModel.create(data, padding, self.config, self.landscape)
            templates = ObservationTemplates.create(data, self.config, tod_structure)
            return None, (model, templates)

        _, (m_stack, t_stack) = jax.lax.scan(build_one, None, self.get_read_indices())
        return m_stack, t_stack

    def accumulate_hits(self, model: ObservationModel) -> Int64[Array, '...']:
        """Accumulate hit count map across all observations."""
        return accumulate_hits(model)  # type: ignore[no-any-return]

    def accumulate_rhs(self, model: ObservationModel) -> StokesPyTreeType:
        """Accumulate the RHS vector across all observations."""
        reader = self.get_reader(['metadata', 'sample_data'])
        read_indices = self.distribute(self.get_padded_read_indices())
        config = self.config
        fill_gaps = config.gaps.fill and not config.binned
        return accumulate_rhs(  # type: ignore[no-any-return]
            model,
            read_indices,
            reader,
            fill_gaps=fill_gaps,
            correlation_length=config.weighting.correlation_length if fill_gaps else None,
            gap_filling_params=config.gaps.fill_options if fill_gaps else None,
        )

    def accumulate_rhs_and_template_partials(
        self,
        model: ObservationModel,
        weight: AbstractLinearOperator,
        explicit: AbstractLinearOperator,
    ) -> tuple[StokesPyTreeType, PyTree]:
        """Streaming pass for ``b = H.T M W_m d`` and the explicit partials ``T.T W_m d``.

        ``weight`` is the per-observation (marginalised) weight ``W_m`` and ``explicit`` the
        per-observation explicit template operator; both carry an obs-stacked leading axis."""
        reader = self.get_reader(['sample_data'])
        read_indices = self.distribute(self.get_padded_read_indices())
        return accumulate_rhs_and_template_partials(  # type: ignore[no-any-return]
            model, weight, explicit, read_indices, reader
        )

    def accumulate_rhs_weighted(
        self, model: ObservationModel, weight: AbstractLinearOperator
    ) -> StokesPyTreeType:
        """Streaming pass for the map RHS ``b = H.T M W_m d`` only (filter+bin, no templates)."""
        reader = self.get_reader(['sample_data'])
        read_indices = self.distribute(self.get_padded_read_indices())
        return accumulate_rhs_weighted(model, weight, read_indices, reader)  # type: ignore[no-any-return]

    def _has_deprojection(self) -> bool:
        """Whether a deprojection is folded into W. Only ATOP; T→P leakage is a fitted
        template (in T), so it does not touch W or the block-diagonality of A."""
        return self.config.method == Methods.ATOP

    def _map_system_block_diagonal(self) -> bool:
        """Whether A = HᵀMWMH is block-diagonal in pixel space, i.e. BJ.I is its exact
        inverse. Requires diagonal-in-time W (no Toeplitz, no folded deprojection) and
        nearest-neighbour pointing (bilinear couples neighbouring pixels)."""
        cfg = self.config
        return (
            cfg.weighting.diagonal_matrix
            and not self._has_deprojection()
            and cfg.pointing.interpolation == 'nearest'
        )

    def get_system_operator(
        self, model: ObservationModel, *, diag: bool = False
    ) -> AbstractLinearOperator:
        H = ScanBlockColumnOperator.create(model.H)
        M = ScanBlockDiagonalOperator.create(model.masker)
        weight = model.diag_W() if diag else model.W
        W = ScanBlockDiagonalOperator.create(weight)
        return (H.T @ M @ W @ M @ H).reduce()

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

    def make_maps_template(self) -> MapMakingResults:
        """Template (destriper-like) estimator over multiple observations.

        Template families are partitioned by their ``explicit`` config flag (see
        :class:`~furax.mapmaking.config.TemplatesConfig`):

        - *explicit* families (``T`` below) have their amplitudes solved in the CG and returned;
        - *marginalised* families (``T_m``) are folded into the weight ``W → W_m`` and never
          explicitly solved (``W_m = W − W T_m (T_mᵀ W T_m)⁻¹ T_mᵀ W``; see :mod:`._marginalize`).

        With no marginalised family ``W_m = W`` (the classic two-step); with no explicit family
        the amplitude CG is skipped entirely (pure filter+bin / ATOP).

        Operators (block-diagonal over observations unless noted; ``H`` is block-column,
        ``H.T`` block-row, so products implicitly sum over observations):

        - ``H`` — pointing/acquisition (sky → TOD), :class:`ScanBlockColumnOperator`
        - ``W_m`` — (marginalised) inverse noise weight, :class:`ScanBlockDiagonalOperator`
        - ``M`` — scanning mask projector, :class:`ScanBlockDiagonalOperator`
        - ``T`` — explicit template operator (amplitudes → TOD), :class:`ScanBlockDiagonalOperator`
        - ``S`` — pixel selector (index operator onto valid pixels)
        - ``A = H.T M W_m M H`` — system matrix
        - ``Z = W_m − W_m M H A⁻¹ H.T M W_m`` — destriper filter

        Equations (with ``W_m`` in place of ``W``):

        - ``b = H.T M W_m d`` — map RHS
        - ``r_T = T.T W_m d − T.T W_m M H A⁻¹ b`` — amplitude system RHS
        - ``(T.T Z T) a = r_T`` — amplitude system (solved via CG). Applied in factored form
          ``T.T Z T = T.T W_m T − (T.T W_m M H) A⁻¹ (H.T M W_m T)`` so each CG matvec streams the
          TOD one observation at a time instead of forming the full per-rank stack.
        - ``map = A⁻¹ (b − H.T M W_m T a)``

        Marginalising makes ``W_m`` non-diagonal in time, so ``A`` is no longer block-diagonal in
        pixel space and the map is solved by PCG (the block-Jacobi inverse becomes a preconditioner).
        """
        if not self.config.binned:
            raise NotImplementedError

        logger_info = lambda msg: self.logger.info(f'MultiObsMapMaker (TwoStep): {msg}')

        _, _, n_pad = self.obs_distribution

        t0 = time.perf_counter()
        t_phase = t0
        model, templates_stack = self.build_model_and_templates()
        model = self.distribute(model.pad(n_pad))
        templates_stack = self.distribute(templates_stack.pad(n_pad))
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logger_info(
            f'Built per-obs models and templates in {time.perf_counter() - t_phase:.1f}s'
            f' (RSS {rss_mb:.0f} MB)'
        )

        with jax.set_mesh(self.mesh):
            t_phase = time.perf_counter()
            hits = accumulate_hits(model)
            jax.block_until_ready(hits)
            logger_info(f'Computed hit map ({time.perf_counter() - t_phase:.1f}s)')

            t_phase = time.perf_counter()
            # only diagonal noise model supported for now
            A = self.get_system_operator(model)
            BJ = BJPreconditioner.create(A)
            icov = BJ.get_blocks().block_until_ready()
            logger_info(
                f'Computed white noise inverse covariance ({time.perf_counter() - t_phase:.1f}s)'
            )

            valid_pixels = self.pixel_selection(hits, icov)
            S = IndexOperator(jnp.where(valid_pixels), in_structure=model.map_structure)

            hits = hits.at[~valid_pixels].set(0)
            icov = jnp.moveaxis(icov, [-2, -1], [0, 1])

            Pinv = (S @ BJ.I @ S.T).reduce()

            map_solver_cfg = self.config.solver
            solver = asdict(map_solver_cfg)

            H = ScanBlockColumnOperator.create(model.H)
            M = ScanBlockDiagonalOperator.create(model.masker)

            # Fold any marginalised templates into the weight: W → W_m (block-diagonal over
            # observations, but no longer diagonal in time). ``W_perobs`` is the per-observation
            # form applied inside the streaming RHS pass; ``W`` is its scan-block wrapper.
            marginal = templates_stack.marginal
            if marginal is not None:
                assert self.config.templates is not None
                W_perobs = build_marginal_weight(
                    model.W, marginal, self.config.templates.regularization, self.mesh
                )
                A_system = (H.T @ M @ ScanBlockDiagonalOperator.create(W_perobs) @ M @ H).reduce()
                block_diagonal = False  # W_m not diagonal in time → A_system not pixel-block-diag
            else:
                W_perobs = model.W
                A_system = A
                block_diagonal = self._map_system_block_diagonal()

            W = ScanBlockDiagonalOperator.create(W_perobs)
            Ht_M_W = (H.T @ M @ W).reduce()  # stacked TOD → sky

            # map-system inverse A⁻¹ (inner, inside Z/RHS and the final map solve). When A is
            # block-diagonal in pixel space, BJ.I is its exact inverse; otherwise (marginalised
            # W_m, Toeplitz W, or pixel-coupling pointing) BJ.I is only a preconditioner → PCG.
            # Same Ainv everywhere → unbiased.
            if block_diagonal:
                Ainv = S.T @ Pinv @ S
            else:
                A_sel = (S @ A_system @ S.T).reduce()
                Ainv = (S.T @ CGInverseOperator(A_sel, preconditioner=Pinv, **solver) @ S).reduce()

            # Explicit templates (whose amplitudes we solve for and return). With none active,
            # marginalisation has absorbed every template and the map is a pure filter+bin solve.
            explicit = templates_stack.operator
            if explicit is None:
                t_phase = time.perf_counter()
                b = self.accumulate_rhs_weighted(model, W_perobs)
                logger_info(f'Built map RHS ({time.perf_counter() - t_phase:.1f}s)')
                logger_info('No explicit templates: skipping amplitude CG (filter+bin)')
                a = None
                amp_steps = 0
                amp_residuals = None
                map_rhs = b
            else:
                T = ScanBlockDiagonalOperator.create(explicit)
                Tt_W_M_H = (T.T @ W @ M @ H).reduce()  # sky → stacked amplitudes (coupling block)

                t_phase = time.perf_counter()
                # Streaming pass: b = H.T M W_m d and r_T_partials = T.T W_m d
                b, r_T = self.accumulate_rhs_and_template_partials(model, W_perobs, explicit)
                logger_info(
                    f'Built map RHS and template RHS partials ({time.perf_counter() - t_phase:.1f}s)'
                )

                t_phase = time.perf_counter()
                # Template system RHS: r_T = T.T W_m d − T.T W_m M H A⁻¹ b
                r_T = furax.tree.sub(r_T, Tt_W_M_H(Ainv(b)))
                logger_info(f'Built amplitude system RHS ({time.perf_counter() - t_phase:.1f}s)')

                # Amplitude system, factored so the matvec never materialises the full stacked
                # TOD: T.T Z T = T.T W_m T − (T.T W_m M H) A⁻¹ (H.T M W_m T)
                #             = Tt_W_T − Tt_W_M_H A⁻¹ Tt_W_M_H.T
                # T.T W_m T reduces to a block-diagonal ScanBlock (one observation at a time); the
                # coupling term acts in the small sky/amplitude spaces via the precomputed block.
                Tt_W_T = (T.T @ W @ T).reduce()
                T_system = (Tt_W_T - Tt_W_M_H @ Ainv @ Tt_W_M_H.T).reduce()

                assert self.config.templates is not None  # two-step requires templates
                amp_solver_cfg = self.config.templates.solver
                logger_info(
                    f'Starting amplitude CG solver'
                    f' (max_steps={amp_solver_cfg.max_steps}, rtol={amp_solver_cfg.rtol})'
                )
                t_phase = time.perf_counter()
                sol_T = furax.linalg.cg(
                    T_system,
                    r_T,
                    **asdict(amp_solver_cfg),
                    iteration_callback=(
                        (
                            lambda i, r: self.logger.debug(
                                f'Amplitude CG iter {int(i)}: ||r||={float(r):.4e}'
                            )
                        )
                        if self.logger.isEnabledFor(logging.DEBUG)
                        else None
                    ),
                )
                a = sol_T.solution
                amp_steps = int(sol_T.num_steps)
                amp_residuals = sol_T.residuals
                logger_info(
                    f'Estimated template amplitudes in {time.perf_counter() - t_phase:.1f}s'
                    f' ({amp_steps} CG iterations,'
                    f' final ||r||='
                    f'{float(amp_residuals[min(amp_steps, amp_solver_cfg.max_steps - 1)]):.4e})'
                )
                # Sky estimates RHS: b − H.T M W_m T a
                map_rhs = furax.tree.sub(b, Ht_M_W(T(a)))

            # Map solve A⁻¹ map_rhs (shared by the explicit and filter+bin paths).
            if block_diagonal:
                t_phase = time.perf_counter()
                estimate = Ainv(map_rhs)
                jax.tree.leaves(estimate)[0].block_until_ready()
                map_steps = 0
                logger_info(
                    f'Estimated map via direct solve ({time.perf_counter() - t_phase:.1f}s)'
                )
            else:
                logger_info(
                    f'Starting map CG solver'
                    f' (max_steps={map_solver_cfg.max_steps}, rtol={map_solver_cfg.rtol})'
                )
                t_phase = time.perf_counter()
                map_sol = furax.linalg.cg(
                    A_sel,
                    S(map_rhs),
                    preconditioner=Pinv,
                    **solver,
                    iteration_callback=(
                        (
                            lambda i, r: self.logger.debug(
                                f'Map CG iter {int(i)}: ||r||={float(r):.4e}'
                            )
                        )
                        if self.logger.isEnabledFor(logging.DEBUG)
                        else None
                    ),
                )
                estimate = S.T(map_sol.solution)
                jax.tree.leaves(estimate)[0].block_until_ready()
                map_steps = int(map_sol.num_steps)
                logger_info(
                    f'Estimated map in {time.perf_counter() - t_phase:.1f}s'
                    f' ({map_steps} CG iterations,'
                    f' final ||r||={float(map_sol.residuals[min(map_steps, map_solver_cfg.max_steps - 1)]):.4e})'
                )

        logger_info(f'Total make_maps_template time: {time.perf_counter() - t0:.1f}s')
        return MapMakingResults(
            map=estimate,
            icov=icov,
            hit_map=hits,
            solver_stats={
                'amplitude': {'num_steps': amp_steps},
                'map': {'num_steps': map_steps},
            },
            cg_residuals=amp_residuals,
            landscape=self.landscape,
            template_amplitudes=(
                None if a is None else {k: jax.tree.map(np.asarray, v) for k, v in a.items()}
            ),
        )

    # Backwards-compatible alias for the renamed two-step estimator.
    make_maps_twostep = make_maps_template

    def _scan_wcs_footprint(self) -> WCSLandscape:
        """Scan observations to determine the combined WCS footprint and build a WCSLandscape.

        Performs a preliminary pass over all observations, loading the pointing data needed
        to compute each observation's sky coverage, then combines them into a unified footprint.
        """
        lc = self.config.landscape
        if lc.wcs is None:
            raise ValueError('WCS landscape config is required for auto footprint scanning.')
        wcs_config = lc.wcs
        res = wcs_config.resolution * pixell.utils.arcmin
        proj = wcs_config.projection.name.lower()
        pointing_fields = ['boresight_quaternions', 'detector_quaternions']

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


@jax.jit
def accumulate_hits(model: ObservationModel) -> Int64[Array, '...']:
    assert isinstance(model.H, CompositionOperator)  # mypy
    pointing = model.H.operands[-1]
    assert isinstance(pointing, PointingOperator)  # mypy
    pointing_i = pointing.as_stokes_i(interpolate=False)

    map_shape = jax.tree.leaves(pointing_i.in_structure)[0].shape

    mesh = jax.sharding.get_abstract_mesh()
    axis = mesh.axis_names[0]

    @jax.shard_map(out_specs=P(), check_vma=False)
    def hits_kernel(pointing_i, masker):  # type: ignore[no-untyped-def]
        per_obs_ones = furax.tree.ones_like(masker.in_structure)

        def step(carry, args):  # type: ignore[no-untyped-def]
            p_i, m_i = args
            masked_i = jax.tree.leaves(m_i(per_obs_ones))[0]
            return carry + jnp.int64(p_i.T(StokesI(masked_i)).i), None

        init = jax.lax.pcast(jnp.zeros(map_shape, jnp.int64), axis, to='varying')
        hits, _ = jax.lax.scan(step, init, (pointing_i, masker))
        return jax.lax.psum(hits, axis_name=axis)

    return hits_kernel(pointing_i, model.masker)  # type: ignore[no-any-return]


@jax.jit(static_argnames=('fill_gaps', 'correlation_length', 'gap_filling_params'))
def accumulate_rhs(
    model: ObservationModel,
    read_indices: Array,
    reader: ObservationReader[T],
    *,
    fill_gaps: bool,
    correlation_length: int | None = None,
    gap_filling_params: GapFillingConfig | None = None,
) -> StokesPyTreeType:
    mesh = jax.sharding.get_abstract_mesh()
    axis = mesh.axis_names[0]

    map_structure = model.map_structure

    @jax.shard_map(out_specs=P(), check_vma=False)
    def rhs_kernel(model: ObservationModel, indices: Array) -> StokesPyTreeType:
        def step(carry: StokesPyTreeType, args: Any) -> tuple[StokesPyTreeType, None]:
            obs, i = args
            data, _ = reader.read(i)
            tod = data['sample_data']
            if fill_gaps:
                if correlation_length is None or gap_filling_params is None:
                    raise ValueError(
                        'fill_gaps=True requires correlation_length and gap_filling_params'
                    )
                N = obs.noise_operator(correlation_length, inverse=False)
                gap_fill = GapFillingOperator(
                    N,
                    obs._get_indexer(),
                    data['metadata'],
                    obs.W,
                    rate=obs.sample_rate,
                    max_cg_steps=gap_filling_params.max_steps,
                    rtol=gap_filling_params.rtol,
                )
                key = jax.random.key(gap_filling_params.seed)
                tod = gap_fill(key, tod)
            return furax.tree.add(carry, obs.H.T(obs.masker(obs.W(tod)))), None

        init = jax.lax.pcast(furax.tree.zeros_like(map_structure), axis, to='varying')
        rhs, _ = jax.lax.scan(step, init, (model, indices))
        return jax.lax.psum(rhs, axis_name=axis)  # type: ignore[no-any-return]

    return rhs_kernel(model, read_indices)


@jax.jit
def accumulate_rhs_and_template_partials(
    model: ObservationModel,
    weight: AbstractLinearOperator,
    explicit: AbstractLinearOperator,
    read_indices: Array,
    reader: ObservationReader[T],
) -> tuple[StokesPyTreeType, PyTree]:
    """Single streaming pass: map RHS ``b = H.T M W_m d`` and explicit stack ``T.T W_m d``.

    ``weight`` and ``explicit`` are per-observation operators (obs-stacked leaves) sliced per
    observation alongside ``model``: ``weight`` applies the marginalised ``W_m`` and
    ``explicit`` projects onto the explicit template amplitudes."""
    mesh = jax.sharding.get_abstract_mesh()
    axis = mesh.axis_names[0]

    map_structure = model.map_structure

    # ``weight`` (the marginalised W_m) may carry scalar leaves with no observation axis — e.g. the
    # ``-1`` of ``W - W T_m G⁻¹ T_mᵀ W``. Split them off so ``scan`` only slices the obs-batched
    # leaves and the scalars are broadcast (closed over), then reassemble per observation.
    w_batched, w_static, w_treedef = partition_obs_leaves(weight, _obs_axis_size(weight))

    @jax.shard_map(out_specs=(P(), P(axis)), check_vma=False)
    def kernel(
        local_model: ObservationModel,
        w_batched: Any,
        w_static: Any,
        local_explicit: AbstractLinearOperator,
        local_indices: Array,
    ) -> tuple[StokesPyTreeType, Any]:
        def step(carry: StokesPyTreeType, args: Any) -> tuple[StokesPyTreeType, Any]:
            model, w_batched_i, explicit, i = args
            weight = combine_obs_leaves(w_treedef, w_batched_i, w_static)
            data, _ = reader.read(i)
            W_d = weight(data['sample_data'])
            Ht_M_W_d = model.H.T(model.masker(W_d))
            Tt_W_d = explicit.T(W_d)
            return furax.tree.add(carry, Ht_M_W_d), Tt_W_d

        init_rhs = jax.lax.pcast(furax.tree.zeros_like(map_structure), axis, to='varying')
        rhs, Tt_W_d = jax.lax.scan(
            step, init_rhs, (local_model, w_batched, local_explicit, local_indices)
        )
        return jax.lax.psum(rhs, axis_name=axis), Tt_W_d

    return kernel(model, w_batched, w_static, explicit, read_indices)


@jax.jit
def accumulate_rhs_weighted(
    model: ObservationModel,
    weight: AbstractLinearOperator,
    read_indices: Array,
    reader: ObservationReader[T],
) -> StokesPyTreeType:
    """Streaming map RHS ``b = H.T M W_m d`` with no template partials (filter+bin path)."""
    mesh = jax.sharding.get_abstract_mesh()
    axis = mesh.axis_names[0]

    map_structure = model.map_structure

    # see accumulate_rhs_and_template_partials: keep scalar (broadcast) leaves of W_m out of scan.
    w_batched, w_static, w_treedef = partition_obs_leaves(weight, _obs_axis_size(weight))

    @jax.shard_map(out_specs=P(), check_vma=False)
    def kernel(
        local_model: ObservationModel,
        w_batched: Any,
        w_static: Any,
        local_indices: Array,
    ) -> StokesPyTreeType:
        def step(carry: StokesPyTreeType, args: Any) -> tuple[StokesPyTreeType, None]:
            model, w_batched_i, i = args
            weight = combine_obs_leaves(w_treedef, w_batched_i, w_static)
            data, _ = reader.read(i)
            W_d = weight(data['sample_data'])
            return furax.tree.add(carry, model.H.T(model.masker(W_d))), None

        init_rhs = jax.lax.pcast(furax.tree.zeros_like(map_structure), axis, to='varying')
        rhs, _ = jax.lax.scan(step, init_rhs, (local_model, w_batched, local_indices))
        return jax.lax.psum(rhs, axis_name=axis)  # type: ignore[no-any-return]

    return kernel(model, w_batched, w_static, read_indices)


def get_obs_distribution_to_process(
    n_obs: int,
    rank: int | None = None,
    n_proc: int | None = None,
    n_local: int | None = None,
) -> tuple[int, int, int]:
    """Compute this process's slice for distributed mapmaking.

    Distributes ``n_obs`` observations across processes as evenly as possible
    (first ``n_obs % n_proc`` processes get one extra), then pads each process's
    share to the next multiple of ``n_local`` so every device has a uniform
    workload.  All processes end up with the same number of total slots
    (``n_owned + n_pad``), which is required for multi-process sharding.

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
    max_owned = base + (1 if remainder > 0 else 0)
    n_per_proc = max_owned + (-max_owned) % n_local  # ceil to next multiple of n_local

    n_owned = base + (1 if rank < remainder else 0)
    start = rank * base + min(rank, remainder)
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
    stokes: ValidStokesType,
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

    def __post_init__(self) -> None:
        return

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
            if isinstance(m, jax.Array) or isinstance(m, np.ndarray):
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
            Methods.TWOSTEP: TwoStepMapmaker,
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
        """Landscape used for mapmaking with given observation"""
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
        """Operator containing pointing information for given observation"""

        det_off_ang = observation.get_detector_offset_angles().astype(landscape.dtype)

        if self.config.pointing.on_the_fly:
            pointing = PointingOperator.create(
                landscape,
                jnp.asarray(observation.get_boresight_quaternions()),
                jnp.asarray(observation.get_detector_quaternions()),
                chunk_size=self.config.pointing.chunk_size,
                interpolate=self.config.pointing.interpolation == 'bilinear',
            )
            return pointing

        else:
            pixel_inds, spin_ang = observation.get_pointing_and_spin_angles(landscape)
            point_ang = spin_ang + det_off_ang[:, None]

            if isinstance(landscape, WCSLandscape | AstropyWCSLandscape):
                assert pixel_inds.shape[-1] == 2, 'Wrong WCS landscape format'
                indexer = IndexOperator(
                    (pixel_inds[..., 0], pixel_inds[..., 1]), in_structure=landscape.structure
                )
            elif isinstance(landscape, HealpixLandscape):
                if pixel_inds.shape[-1] == 1:
                    pixel_inds = pixel_inds[..., 0]
                indexer = IndexOperator(pixel_inds, in_structure=landscape.structure)

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
        """Acquisition operator mapping sky maps to time-ordered data"""
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
        """Flag operator which selects only the scanning intervals
        of the given TOD of shape (ndets, nsamps).
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
        """Flag operator which sets the values outside the scanning intervals
        of the given TOD (of shape (ndets, nsamps)) to zero.
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
        """Flag operator which sets the values of the given TOD (of shape (ndets, nsamps)) to
        zero at masked (flagged) samples.
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
        """Return a noise model for the observation, corresponding to
        the type (diagonal, toeplitz, ...) specified by the mapmaker.
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
            jnp.asarray(observation.get_tods()),
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
        """Select indices of map pixels satisfying
        the minimum fractional hits (hits_cut) and condition number (cond_cut) criteria"""
        config = self.config

        # eigs = jnp.linalg.eigvalsh(blocks)
        eigs = np.linalg.eigvalsh(blocks)
        hits_quantile = np.quantile(eigs[(eigs[..., -1] > 0),], q=0.95)
        valid = jnp.logical_and(
            eigs[..., -1] > config.hits_cut * hits_quantile,
            eigs[..., 0] > config.cond_cut * eigs[..., -1],
        )
        return IndexOperator(jnp.where(valid), in_structure=landscape.structure)

    def get_template_operator(
        self, observation: AbstractGroundObservation[Any]
    ) -> BlockRowOperator:
        """Create and return a template operator corresponding to the
        name and configuration provided.
        """
        config = self.config
        assert config.templates is not None
        blocks: dict[str, AbstractLinearOperator] = {}

        if poly := config.templates.polynomial:
            blocks['polynomial'] = templates.PolynomialTemplateOperator.create(
                max_poly_order=poly.legendre.max_order,
                intervals=observation.get_scanning_intervals(),
                times=jnp.asarray(observation.get_elapsed_times()),
                n_dets=observation.n_detectors,
                dtype=config.dtype,
            )
        if sss := config.templates.scan_synchronous:
            blocks['scan_synchronous'] = templates.ScanSynchronousTemplateOperator.create(
                min_poly_order=sss.legendre.min_order,
                max_poly_order=sss.legendre.max_order,
                azimuth=jnp.asarray(observation.get_azimuth()),
                n_dets=observation.n_detectors,
                dtype=config.dtype,
            )
        if hwpss := config.templates.hwp_synchronous:
            blocks['hwp_synchronous'] = templates.HWPSynchronousTemplateOperator.create(
                n_harmonics=hwpss.n_harmonics,
                hwp_angles=jnp.asarray(observation.get_hwp_angles()),
                n_dets=observation.n_detectors,
                dtype=config.dtype,
            )
        if azhwpss := config.templates.azhwp_synchronous:
            azimuth = jnp.asarray(observation.get_azimuth())
            hwp_angles = jnp.asarray(observation.get_hwp_angles())
            if azhwpss.split_scans:
                blocks['azhwp_synchronous_left'] = (
                    templates.AzimuthHWPSynchronousTemplateOperator.create(
                        n_polynomials=azhwpss.legendre.max_order - azhwpss.legendre.min_order + 1,
                        n_harmonics=azhwpss.n_harmonics,
                        azimuth=azimuth,
                        hwp_angles=hwp_angles,
                        n_dets=observation.n_detectors,
                        dtype=config.dtype,
                        scan_mask=jnp.asarray(observation.get_left_scan_mask()),
                    )
                )
                blocks['azhwp_synchronous_right'] = (
                    templates.AzimuthHWPSynchronousTemplateOperator.create(
                        n_polynomials=azhwpss.legendre.max_order - azhwpss.legendre.min_order + 1,
                        n_harmonics=azhwpss.n_harmonics,
                        azimuth=azimuth,
                        hwp_angles=hwp_angles,
                        n_dets=observation.n_detectors,
                        dtype=config.dtype,
                        scan_mask=jnp.asarray(observation.get_right_scan_mask()),
                    )
                )
            else:
                blocks['azhwp_synchronous'] = (
                    templates.AzimuthHWPSynchronousTemplateOperator.create(
                        n_polynomials=azhwpss.legendre.max_order - azhwpss.legendre.min_order + 1,
                        n_harmonics=azhwpss.n_harmonics,
                        azimuth=azimuth,
                        hwp_angles=hwp_angles,
                        n_dets=observation.n_detectors,
                        dtype=config.dtype,
                    )
                )
        if binazhwpss := config.templates.binazhwp_synchronous:
            blocks['binazhwp_synchronous'] = (
                templates.BinAzimuthHWPSynchronousTemplateOperator.create(
                    n_azimuth_bins=binazhwpss.bins.n_bins,
                    n_harmonics=binazhwpss.n_harmonics,
                    interpolate_azimuth=binazhwpss.bins.interpolate,
                    smooth_interpolation=binazhwpss.bins.smooth,
                    azimuth=jnp.asarray(observation.get_azimuth()),
                    hwp_angles=jnp.asarray(observation.get_hwp_angles()),
                    n_dets=observation.n_detectors,
                    dtype=config.dtype,
                )
            )
        if ground := config.templates.ground:
            azimuth = jnp.asarray(observation.get_azimuth())
            elevation = jnp.asarray(observation.get_elevation())
            detector_quaternions = jnp.asarray(observation.get_detector_quaternions())
            self._ground_landscape = templates.GroundTemplateOperator.get_landscape(
                azimuth_resolution=ground.azimuth_resolution,
                elevation_resolution=ground.elevation_resolution,
                boresight_azimuth=azimuth,
                boresight_elevation=elevation,
                detector_quaternions=detector_quaternions,
                stokes='IQU',
                dtype=config.dtype,
            )
            ground_op = templates.GroundTemplateOperator.create(
                azimuth_resolution=ground.azimuth_resolution,
                elevation_resolution=ground.elevation_resolution,
                boresight_azimuth=azimuth,
                boresight_elevation=elevation,
                boresight_rotation=jnp.zeros_like(azimuth),
                detector_quaternions=detector_quaternions,
                hwp_angles=jnp.asarray(observation.get_hwp_angles()),
                stokes='IQU',
                dtype=config.dtype,
                landscape=self._ground_landscape,
                chunk_size=config.pointing.chunk_size,
            )
            ones_tod = jnp.ones((observation.n_detectors, observation.n_samples), dtype=jnp.float64)
            self._ground_coverage = ground_op.T(ones_tod)
            nonzero_hits = jnp.argwhere(self._ground_coverage.i > 0)
            indexer = IndexOperator(
                (nonzero_hits[:, 0], nonzero_hits[:, 1]),
                in_structure=furax.tree.as_structure(self._ground_coverage),
            )
            flattener = templates.StokesIQUFlattenOperator(in_structure=indexer.out_structure)
            self._ground_selector = flattener @ indexer

            blocks['ground'] = ground_op @ self._ground_selector.T

        return BlockRowOperator(blocks=blocks)


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
        weights = np.array(system.get_blocks())

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
    """Class for mapmaking with maximum likelihood (ML) estimator"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validation on config
        if self.config.binned:
            raise ValueError('ML Mapmaker is incompatible with binned=True')
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
        if config.weighting.mode == WeightingMode.IDENTITY:
            diag_inv_noise = inv_noise
        elif isinstance(inv_noise, SymmetricBandToeplitzOperator):
            diag_inv_noise = DiagonalOperator(
                inv_noise.band_values[..., [0]], in_structure=data_struct
            )
        else:
            raise NotImplementedError
        diag_system = BJPreconditioner.create(acquisition.T @ diag_inv_noise @ masker @ acquisition)
        logger_info('Created approximate system matrix')

        # Map pixel selection
        blocks = diag_system.get_blocks()
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

        # Templates (optional)
        if config.use_templates:
            template_op = self.get_template_operator(observation)
            logger_info('Built template operators')
            REGVAL = config.templates.regularization  # type: ignore[union-attr]
            tmpl_inv_sys = {}
            regs = {}
            for tmpl, tmpl_op in template_op.blocks.items():
                tmpl_sys = (tmpl_op.T @ diag_inv_noise @ masker @ tmpl_op).reduce()
                # Approximation to the diagonal of the matrix
                norm_sys = jnp.abs(jax.jit(tmpl_sys)(furax.tree.ones_like(tmpl_op.in_structure)))
                # Regualrisation value is REGVAL times the smallest non-zero eigenvalue
                regs[tmpl] = REGVAL * jnp.min(norm_sys[norm_sys > 0])
                tmpl_inv_sys[tmpl] = DiagonalOperator(
                    (norm_sys + regs[tmpl]),
                    in_structure=tmpl_op.in_structure,
                ).inverse()
            template_preconditioner = BlockDiagonalOperator(tmpl_inv_sys)
            logger_info('Built template preconditioner')
            template_reg_op = BlockDiagonalOperator(
                [
                    DiagonalOperator(jnp.array([0.0]), in_structure=selector.out_structure),
                    {
                        tmpl: regs[tmpl]
                        * IdentityOperator(in_structure=template_op.blocks[tmpl].in_structure)
                        for tmpl in template_op.blocks.keys()
                    },
                ]
            )
            logger_info('Built template regularizer')
            print(f'Template operator input structure: {template_op.in_structure}')

        # Mapmaking operator
        p: AbstractLinearOperator
        h: AbstractLinearOperator
        if config.use_templates:
            p = BlockDiagonalOperator([preconditioner, template_preconditioner])
            h = BlockRowOperator([acquisition @ selector.T, template_op])
            reg = template_reg_op
        else:
            p = preconditioner
            h = acquisition @ selector.T

        if not config.gaps.nested_pcg:
            M = masker @ inv_noise @ masker
        else:
            nested_solver = lineax.CG(
                rtol=config.solver.rtol,
                atol=config.solver.atol,
                max_steps=30,
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

        solver = lineax.CG(**asdict(config.solver))
        options = {'solver': solver, 'preconditioner': p}
        if config.use_templates:
            mapmaking_operator = (h.T @ M @ h + reg).I(**options) @ h.T @ M
        else:
            mapmaking_operator = (h.T @ M @ h).I(**options) @ h.T @ M

        @jax.jit
        def process(d):  # type: ignore[no-untyped-def]
            return mapmaking_operator.reduce()(d)

        logger_info('Completed setting up the solver')

        # Run mapmaking
        if config.use_templates:
            rec_map, tmpl_ampl = process(data)
        else:
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
        if config.use_templates:
            for key in tmpl_ampl.keys():
                output[f'template_{key}'] = tmpl_ampl[key]
                output[f'template_reg_{key}'] = np.array(regs[key])
                aux_data = template_op.blocks[key].compute_auxiliary_data(tmpl_ampl[key])
                for aux_key in aux_data.keys():
                    output[f'template_{key}_{aux_key}'] = aux_data[aux_key]
            if 'ground' in tmpl_ampl.keys():
                output['ground_landscape'] = self._ground_landscape
                output['ground_coverage'] = self._ground_coverage
                output['ground_map'] = self._ground_selector.T(tmpl_ampl['ground'])
        if config.debug:
            proj_map = (masker @ acquisition)(result_map)
            if config.use_templates:
                projs = {
                    'proj_map': proj_map,
                    **{
                        f'proj_{tmpl}': (masker @ template_op.blocks[tmpl])(tmpl_ampl[tmpl])
                        for tmpl in tmpl_ampl
                    },
                }
            else:
                projs = {'proj_map': proj_map}
            output['projs'] = projs

        return output


class TwoStepMapmaker(MapMaker):
    """Class for binned mapmaking with templates, using the two-step estimation method"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validation on config
        if not self.config.binned:
            raise ValueError('Two-Step Mapmaker is incompatible with binned=False')
        if self.config.demodulated:
            raise ValueError('Two-Step Mapmaker is incompatible with demodulated=True')
        if not self.config.use_templates:
            raise ValueError('Two-Step Mapmaker is incompatible with no templates')

    def make_map(self, observation: AbstractGroundObservation[Any]) -> dict[str, Any]:
        config = self.config
        logger_info = lambda msg: self.logger.info(f'Two-Step Mapmaker: {msg}')

        # Data and landscape
        data = jnp.asarray(observation.get_tods(), dtype=config.dtype)
        data_struct = ShapeDtypeStruct(data.shape, data.dtype)
        landscape = self.get_landscape(observation)

        # Acquisition (I, Q, U Maps -> TOD)
        acquisition = self.get_acquisition(observation, landscape=landscape)
        logger_info('Created acquisition operator')

        # Optional mask for scanning
        masker = self.get_scanning_mask_projector(observation)
        logger_info('Created scanning mask operator')

        # Noise
        noise_model = self.get_or_fit_noise_model(observation)
        inv_noise = noise_model.inverse_operator(data_struct)
        logger_info('Created inverse noise covariance operator')

        # System matrix
        system = BJPreconditioner.create(acquisition.T @ masker @ inv_noise @ masker @ acquisition)
        logger_info('Created system matrix')

        # Map pixel selection
        blocks = system.get_blocks()
        selector = self.get_pixel_selector(blocks, landscape)
        logger_info(
            f'Selected {prod(selector.out_structure.shape)}\
                            /{prod(landscape.shape)} pixels'
        )

        # Templates
        template_op = self.get_template_operator(observation)
        logger_info('Built template operators')

        # Define operators
        system_inv = selector @ system.inverse() @ selector.T
        A = acquisition @ selector.T
        M = inv_noise
        mp = masker
        FA = M - M @ mp @ A @ system_inv @ A.T @ mp @ M

        solver = lineax.CG(**asdict(config.solver))
        with Config(solver=solver):
            template_estimator = (
                (template_op.T @ mp @ FA @ mp @ template_op).I @ template_op.T @ mp @ FA @ mp
            )
        map_estimator = system_inv @ A.T @ mp @ M @ mp

        @jax.jit
        def process(d):  # type: ignore[no-untyped-def]
            x = template_estimator(d)  # Template amplitude estimates
            s = map_estimator(d - template_op(x))  # Map estimates
            return s, x

        logger_info('Completed setting up the solver')

        # Run mapmaking
        rec_map, tmpl_ampl = process(data)
        result_map = selector.T(rec_map)
        result_map.i.block_until_ready()
        logger_info('Finished mapmaking computation')

        # Format output and compute auxiliary data
        final_map = np.array([result_map.i, result_map.q, result_map.u])

        output = {'map': final_map, 'weights': blocks}
        for key in tmpl_ampl.keys():
            output[f'template_{key}'] = tmpl_ampl[key]
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
            proj_map = (mp @ acquisition)(result_map)
            projs = {
                'proj_map': proj_map,
                **{
                    f'proj_{tmpl}': (mp @ template_op.blocks[tmpl])(tmpl_ampl[tmpl])
                    for tmpl in tmpl_ampl
                },
            }
            output['projs'] = projs

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
        atop_projector = templates.ATOPProjectionOperator(
            self.config.atop_tau, in_structure=data_struct
        )

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
        blocks = diag_system.get_blocks()
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

        solver = lineax.CG(**asdict(self.config.solver))
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
    """Class that adds the input Stokes signals to a single HWP-modulated signal
    Similar to LinearPolarizerOperator @ QURotationOperator(hwp_angle), except that
    only half of the QU rotation needs to be computed
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

    def mv(self, x: StokesPyTreeType) -> Float[Array, '...']:
        return x.i + self.cos_hwp_angle[None, :] * x.q + self.sin_hwp_angle[None, :] * x.u  # type: ignore[union-attr]


class QUModulationOperator(AbstractLinearOperator):
    """Class that adds the input Stokes signals to a single HWP-modulated signal
    Similar to LinearPolarizerOperator @ QURotationOperator(hwp_angle), except that
    only half of the QU rotation needs to be computed
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

    def mv(self, x: StokesPyTreeType) -> Float[Array, '...']:
        return self.cos_hwp_angle[None, :] * x.q + self.sin_hwp_angle[None, :] * x.u  # type: ignore[union-attr]
