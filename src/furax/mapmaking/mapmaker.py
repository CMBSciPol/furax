import pickle
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from logging import Logger
from math import prod
from pathlib import Path
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pixell.enmap
import pixell.utils
from astropy.io import fits
from astropy.wcs import WCS
from jax import ShapeDtypeStruct
from jaxtyping import Array, Bool, DTypeLike, Float, Int64, Integer

import furax.linalg
import furax.tree as tree
from furax import (
    AbstractLinearOperator,
    Config,
    DiagonalOperator,
    IdentityOperator,
    MaskOperator,
    OperatorTag,
    SymmetricBandToeplitzOperator,
)
from furax.core import BlockDiagonalOperator, BlockRowOperator, IndexOperator
from furax.interfaces.lineax import as_lineax_operator
from furax.obs.landscapes import (
    AstropyWCSLandscape,
    HealpixLandscape,
    StokesLandscape,
    WCSLandscape,
)
from furax.obs.operators import HWPOperator, LinearPolarizerOperator, QURotationOperator
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, StokesIQU, StokesPyTreeType, ValidStokesType

from . import templates
from ._geometry import minimum_enclosing_arc
from ._logger import logger as furax_logger
from ._model import (
    MultiObsTemplateOperator,
    ObservationModel,
    ObservationTemplates,
    SystemOperator,
)
from ._observation import AbstractGroundObservation, AbstractLazyObservation
from ._reader import ObservationReader
from .config import LandscapeConfig, MapMakingConfig, Methods, WCSConfig
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel
from .preconditioner import BJPreconditioner
from .results import MapMakingResults
from .templates import template_required_fields

T = TypeVar('T')


def _accumulate_over_obs(
    fn: Callable[..., Any],
    init: Any,
    *xs: Any,
) -> Any:
    """Sum ``fn(*per_obs_slice_of_xs)`` over observations via tree.add."""

    def step(carry: Any, args: Any) -> tuple[Any, None]:
        return tree.add(carry, fn(*args)), None

    out, _ = jax.lax.scan(step, init, xs)
    return out


def _stack_over_obs(
    fn: Callable[..., Any],
    *xs: Any,
) -> Any:
    """Stack ``fn(*per_obs_slice_of_xs)`` along the leading observation axis."""

    def step(_: Any, args: Any) -> tuple[None, Any]:
        return None, fn(*args)

    _, out = jax.lax.scan(step, None, xs)
    return out


def _accumulate_and_stack_over_obs(
    fn: Callable[..., tuple[Any, Any]],
    init: Any,
    *xs: Any,
) -> tuple[Any, Any]:
    """Fused variant: ``fn`` returns ``(contribution, slice)`` per obs.

    Sums the contributions into ``init`` and stacks the slices along the leading axis.
    """

    def step(carry: Any, args: Any) -> tuple[Any, Any]:
        contrib, slc = fn(*args)
        return tree.add(carry, contrib), slc

    return jax.lax.scan(step, init, xs)


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
        if self.config.method == Methods.ATOP:
            if not self.config.binned:
                raise ValueError('ATOP requires a white noise model (noise.white=True).')
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
        elif self.config.method == Methods.TWOSTEP:
            if not self.config.binned:
                raise ValueError('TwoStep requires a white noise model (noise.white=True).')
            if self.config.demodulated:
                raise ValueError('TwoStep is incompatible with demodulated data.')
            if not self.config.use_templates:
                raise ValueError('TwoStep requires at least one active template.')

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

    def run(self, out_dir: str | Path | None = None) -> MapMakingResults:
        """Runs the mapmaker and return results after saving them to the given directory."""
        results = self.make_maps()

        # Save outputs
        if out_dir is not None:
            out_dir = Path(out_dir)
            results.save(out_dir)
            self.logger.info(f'saved results to {out_dir}')
            self.config.dump_yaml(out_dir / 'mapmaking_config.yaml')
            self.logger.info('saved mapmaking configuration to file')

        return results

    def get_reader(self, data_field_names: list[str]) -> ObservationReader[T]:
        """Returns a reader for a list of requested fields."""
        return ObservationReader(
            self.observations,
            requested_fields=data_field_names,
            demodulated=self.config.demodulated,
            stokes=self.config.landscape.stokes,
        )

    def make_maps(self) -> MapMakingResults:
        """Computes the mapmaker results (maps and other products)."""
        if self.config.method == Methods.TWOSTEP:
            return self.make_maps_twostep()

        logger_info = lambda msg: self.logger.info(f'MultiObsMapMaker: {msg}')

        model = self.build_model()
        logger_info('Created system operator')

        selector, hits, icov, BJ = self._prepare_pixel_system(model, logger_info)
        apply_S_inv = self._make_S_inv_solver(model, selector, BJ)

        rhs = self.accumulate_rhs(model)
        logger_info('Accumulated RHS vector')

        estimate, stats = apply_S_inv(rhs)
        logger_info(f'Finished mapmaking (iteration steps: {stats["num_steps"]})')
        return MapMakingResults(
            map=estimate,
            icov=icov,
            hit_map=hits,
            solver_stats=stats,
            landscape=self.landscape,
        )

    def _prepare_pixel_system(
        self,
        model: ObservationModel,
        logger_info: Any,
    ) -> tuple[IndexOperator, Integer[Array, ' pixels'], Float[Array, '...'], BJPreconditioner]:
        """Common prelude: hit map, icov, pixel selection, BJ preconditioner."""
        A = SystemOperator(model)
        hits = self.accumulate_hits(model).block_until_ready()
        logger_info('Computed hit map')

        sysdiag = A if self.config.binned else SystemOperator(model, diag=True)
        BJ = BJPreconditioner.create(sysdiag)
        icov = BJ.get_blocks().block_until_ready()
        logger_info('Computed white noise inverse covariance')

        valid_pixels = self.pixel_selection(hits, icov)
        selector = IndexOperator(jnp.where(valid_pixels), in_structure=model.map_structure)
        n_selected = jnp.sum(valid_pixels)
        n_observed = jnp.sum(hits > 0)
        n_total = valid_pixels.size
        logger_info(f'Selected {n_selected} pixels ({n_observed} seen, {n_total} total)')

        hits = hits.at[~valid_pixels].set(0)  # excluded pixels have zero hits
        icov = jnp.moveaxis(icov, [-2, -1], [0, 1])  # (*pixels, ns, ns) → (ns, ns, *pixels)
        return selector, hits, icov, BJ

    def _make_S_inv_solver(
        self,
        model: ObservationModel,
        selector: IndexOperator,
        BJ: BJPreconditioner,
    ) -> Callable[[StokesPyTreeType], tuple[StokesPyTreeType, dict[str, Any]]]:
        """Return a closure ``apply(rhs_map) → (map_estimate, solver_stats)``.

        Applies the (selector-reduced) inverse of ``SystemOperator(model)`` via PCG with the
        BJ preconditioner — the same kernel used by both the single-step and the two-step
        map estimators.
        """
        A = SystemOperator(model)
        solver = lx.CG(**asdict(self.config.solver))
        spd = OperatorTag.POSITIVE_SEMIDEFINITE
        lx_system = as_lineax_operator(selector @ A @ selector.T, spd)
        M = (selector @ BJ.I @ selector.T).reduce()
        lx_precond = as_lineax_operator(M, spd)

        def apply_S_inv(
            rhs_map: StokesPyTreeType,
        ) -> tuple[StokesPyTreeType, dict[str, Any]]:
            rhs_reduced = selector(rhs_map)
            y0 = M(rhs_reduced)
            sol = lx.linear_solve(
                lx_system,
                rhs_reduced,
                solver=solver,
                options={'preconditioner': lx_precond, 'y0': y0},
                throw=False,
            )
            return selector.T(sol.value), sol.stats

        return apply_S_inv

    def make_maps_twostep(self) -> MapMakingResults:
        """Two-step (destriper-like) estimator over multiple observations.

        Solves the template amplitudes from the filtered system and then the map
        from the template-cleaned data. Template amplitudes are returned per obs
        in ``MapMakingResults.template_amplitudes``.

        The time-ordered data is never held entirely in memory: each access streams
        one observation at a time through the reader. A single streaming pass over the
        TOD is needed — both the standard map RHS and the per-obs template partials are
        built in that pass; the template-cleaned map RHS is then assembled in template
        space as ``rhs_map_d − H.T mp W T x_T`` without re-reading the TOD.

        Math (single-mp convention, matches :meth:`accumulate_rhs`; ``mp`` is block-
        diagonal over observations, ``W`` and ``T`` likewise, ``H`` is block-column,
        ``H.T`` block-row, so the matrix products implicitly sum over observations):

        - ``S = H.T mp W mp H`` (existing :class:`SystemOperator`)
        - ``FA = W − W mp H S^{-1} H.T mp W``
        - ``rhs_map_d = H.T mp W d`` (standard RHS, "P.T W d")
        - ``corr_map_d = S^{-1} rhs_map_d``
        - ``rhs_T = T.T W (d − mp H corr_map_d)``
        - ``sys_T(x) = T.T W (T x − mp H S^{-1} H.T mp W T x)``
        - ``map = S^{-1} H.T mp W (d − T x_T)``
        """
        logger_info = lambda msg: self.logger.info(f'MultiObsMapMaker (TwoStep): {msg}')
        n_obs = len(self.observations)

        # Build stacked model and templates
        model = self.build_model()
        templates_stack = self.build_templates()
        logger_info('Built per-obs models and templates')

        # Prelude shared with make_maps
        selector, hits, icov, BJ = self._prepare_pixel_system(model, logger_info)
        S_inv = self._make_S_inv_solver(model, selector, BJ)

        # Streaming pass 1: fused map RHS (P.T W d) and per-obs template partials (T.T W d)
        data_reader = self.get_reader(['sample_data'])
        init_map = tree.zeros_like(model.map_structure)

        def rhs_pass_one(i, m, t_obs):  # type: ignore[no-untyped-def]
            data, _ = data_reader.read(i)
            W_d = m.W(data['sample_data'])
            return m.H.T(m.masker(W_d)), t_obs.obs_template_op.T(W_d)

        rhs_map_d, tmpl_partials_d = _accumulate_and_stack_over_obs(
            rhs_pass_one,
            init_map,
            jnp.arange(data_reader.count),
            model,
            templates_stack,
        )
        logger_info('Built map RHS and template RHS partials (single streaming pass over TOD)')

        # corr_map_d = S^{-1}(rhs_map_d)
        corr_map_d, _ = S_inv(rhs_map_d)

        # Per-obs (no d): T.T W mp H corr_map_d  →  stacked template-domain correction
        tmpl_corrections_d = _stack_over_obs(
            lambda m, t_obs: t_obs.obs_template_op.T(m.W(m.masker(m.H(corr_map_d)))),
            model,
            templates_stack,
        )
        rhs_T = jax.tree.map(jnp.subtract, tmpl_partials_d, tmpl_corrections_d)
        logger_info('Built amplitude system RHS')

        # Amplitude system sys_T(x) — does not touch the TOD.
        def sys_T_mv(x):  # type: ignore[no-untyped-def]
            """T.T FA T applied to stacked amplitudes."""

            def step_one(m, t_obs, x_i):  # type: ignore[no-untyped-def]
                W_Tx = m.W(t_obs.obs_template_op(x_i))
                return m.H.T(m.masker(W_Tx)), t_obs.obs_template_op.T(W_Tx)

            rhs_map_x, partials_x = _accumulate_and_stack_over_obs(
                step_one, init_map, model, templates_stack, x
            )
            corr_map_x, _ = S_inv(rhs_map_x)
            corrections_x = _stack_over_obs(
                lambda m, t_obs: t_obs.obs_template_op.T(m.W(m.masker(m.H(corr_map_x)))),
                model,
                templates_stack,
            )
            return jax.tree.map(jnp.subtract, partials_x, corrections_x)

        # Amplitude preconditioner: per-template per-obs diagonal of (T.T W mp T)
        regval = self.config.templates.regularization  # type: ignore[union-attr]

        def per_obs_precond_diag(m, t_obs):  # type: ignore[no-untyped-def]
            out = {}
            for name, blk in t_obs.obs_template_op.blocks.items():
                tmpl_sys = (blk.T @ m.W @ m.masker @ blk).reduce()
                out[name] = jnp.abs(tmpl_sys(tree.ones_like(blk.in_structure)))
            return out

        diag_stacked = _stack_over_obs(per_obs_precond_diag, model, templates_stack)
        regs = jax.tree.map(
            lambda d: regval * jnp.min(jnp.where(d > 0, d, jnp.inf)),
            diag_stacked,
        )
        precond_diag = jax.tree.map(lambda d, r: 1.0 / (d + r), diag_stacked, regs)

        def precond_amp_mv(x):  # type: ignore[no-untyped-def]
            return jax.tree.map(lambda d, xi: d * xi, precond_diag, x)

        # Outer CG: solve sys_T x = rhs_T in amplitude space
        amp_struct = MultiObsTemplateOperator(templates_stack, n_obs).in_structure
        lx_sys_T = lx.FunctionLinearOperator(sys_T_mv, amp_struct, lx.positive_semidefinite_tag)
        lx_precond_T = lx.FunctionLinearOperator(
            precond_amp_mv, amp_struct, lx.positive_semidefinite_tag
        )
        outer_solver = lx.CG(**asdict(self.config.solver))
        sol_T = lx.linear_solve(
            lx_sys_T,
            rhs_T,
            solver=outer_solver,
            options={'preconditioner': lx_precond_T},
            throw=False,
        )
        x_T = sol_T.value
        logger_info(f'Solved amplitude system (iteration steps: {sol_T.stats["num_steps"]})')

        # Final map RHS — no second TOD pass: map_rhs = rhs_map_d − H.T mp W T x_T
        tmpl_map_contrib = _accumulate_over_obs(
            lambda m, t_obs, xT_i: m.H.T(m.masker(m.W(t_obs.obs_template_op(xT_i)))),
            init_map,
            model,
            templates_stack,
            x_T,
        )
        map_rhs = jax.tree.map(jnp.subtract, rhs_map_d, tmpl_map_contrib)
        estimate, map_stats = S_inv(map_rhs)
        jax.tree.leaves(estimate)[0].block_until_ready()
        logger_info(f'Finished map estimation (iteration steps: {map_stats["num_steps"]})')

        return MapMakingResults(
            map=estimate,
            icov=icov,
            hit_map=hits,
            solver_stats={'amplitude': sol_T.stats, 'map': map_stats},
            landscape=self.landscape,
            template_amplitudes={k: np.array(v) for k, v in x_T.items()},
        )

    def build_model(self) -> ObservationModel:
        reader = self.get_reader(sorted(self._model_required_fields()))

        def build_one(_, i):  # type: ignore[no-untyped-def]
            data, padding = reader.read(i)
            return None, ObservationModel.create(data, padding, self.config, self.landscape)

        _, model = jax.lax.scan(build_one, None, jnp.arange(reader.count))
        return model  # type: ignore[no-any-return]

    def build_templates(self) -> ObservationTemplates:
        """Build a stacked :class:`ObservationTemplates` from each observation."""
        assert self.config.use_templates, 'build_templates called without active templates'
        assert self.config.templates is not None
        # Reader needs the template fields plus the acquisition basics for tod_structure shape
        fields = self._model_required_fields() | template_required_fields(self.config.templates)
        reader = self.get_reader(sorted(fields))

        # Reader pads each observation to a common (max n_dets, max n_samples). Use that
        # padded shape so the template operator's static dimensions match the data leaves
        # extracted from the stacked model.
        tod_ref = reader.out_structure['sample_data']
        tod_structure = jax.ShapeDtypeStruct(tod_ref.shape, self.config.dtype)

        def build_one(_, i):  # type: ignore[no-untyped-def]
            data, _ = reader.read(i)
            return None, ObservationTemplates.create(data, self.config, tod_structure)

        _, templates_stack = jax.lax.scan(build_one, None, jnp.arange(reader.count))
        return templates_stack  # type: ignore[no-any-return]

    def _model_required_fields(self) -> set[str]:
        """Reader fields required to build the per-observation :class:`ObservationModel`."""
        fields = {
            'boresight_quaternions',
            'detector_quaternions',
            'valid_sample_masks',
            'timestamps',
        }
        if not self.config.demodulated:
            # FIXME: this does not handle the case of a telescope without HWP
            fields.add('hwp_angles')
        if self.config.scanning_mask:
            fields.add('valid_scanning_masks')
        if self.config.noise.fit_from_data:
            fields.update({'sample_data', 'hwp_angles'})
        else:
            fields.add('noise_model_fits')
        if self.config.gaps.fill and not self.config.binned:
            fields.add('metadata')
        return fields

    def accumulate_hits(self, models: ObservationModel) -> Int64[Array, ' pixels']:
        def acc(carry, model):  # type: ignore[no-untyped-def]
            return carry + model.hits(), None

        init = jnp.zeros(self.landscape.shape, dtype=jnp.int64)
        total, _ = jax.lax.scan(acc, init, models)
        return total

    def accumulate_rhs(self, models: ObservationModel) -> StokesPyTreeType:
        """Accumulate the RHS vector across all observations"""
        reader = self.get_reader(['metadata', 'sample_data'])

        def acc(carry, args):  # type: ignore[no-untyped-def]
            i, model = args
            data, _ = reader.read(i)
            carry = carry + model.rhs(data, self.config)
            return carry, None

        init = tree.zeros_like(models.map_structure)
        total, _ = jax.lax.scan(acc, init, (jnp.arange(reader.count), models))
        return total  # type: ignore[no-any-return]

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
                observation.get_boresight_quaternions(),
                observation.get_detector_quaternions(),
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
                angles=observation.get_detector_offset_angles().astype(self.config.dtype)[:, None],
            )
            hwp = HWPOperator.create(
                **meta,  # type: ignore[arg-type]
                angles=observation.get_hwp_angles().astype(self.config.dtype),
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
        Model = WhiteNoiseModel if config.binned else AtmosphericNoiseModel

        if not config.noise.fit_from_data:
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
            observation.get_tods(), fs=observation.sample_rate, nperseg=config.noise.fitting.nperseg
        )
        hwp_frequency = observation.get_hwp_frequency()
        return Model.fit_psd_model(
            f,
            Pxx,
            sample_rate=jnp.array(observation.sample_rate),
            hwp_frequency=hwp_frequency,
            config=config.noise.fitting,
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
                max_poly_order=poly.max_poly_order,
                intervals=observation.get_scanning_intervals(),
                times=observation.get_elapsed_times(),
                n_dets=observation.n_detectors,
                dtype=config.dtype,
            )
        if sss := config.templates.scan_synchronous:
            blocks['scan_synchronous'] = templates.ScanSynchronousTemplateOperator.create(
                min_poly_order=sss.min_poly_order,
                max_poly_order=sss.max_poly_order,
                azimuth=jnp.array(observation.get_azimuth()),
                n_dets=observation.n_detectors,
                dtype=config.dtype,
            )
        if hwpss := config.templates.hwp_synchronous:
            blocks['hwp_synchronous'] = templates.HWPSynchronousTemplateOperator.create(
                n_harmonics=hwpss.n_harmonics,
                hwp_angles=observation.get_hwp_angles(),
                n_dets=observation.n_detectors,
                dtype=config.dtype,
            )
        if azhwpss := config.templates.azhwp_synchronous:
            if azhwpss.split_scans:
                blocks['azhwp_synchronous_left'] = (
                    templates.AzimuthHWPSynchronousTemplateOperator.create(
                        n_polynomials=azhwpss.n_polynomials,
                        n_harmonics=azhwpss.n_harmonics,
                        azimuth=observation.get_azimuth(),
                        hwp_angles=observation.get_hwp_angles(),
                        n_dets=observation.n_detectors,
                        dtype=config.dtype,
                        scan_mask=observation.get_left_scan_mask(),
                    )
                )
                blocks['azhwp_synchronous_right'] = (
                    templates.AzimuthHWPSynchronousTemplateOperator.create(
                        n_polynomials=azhwpss.n_polynomials,
                        n_harmonics=azhwpss.n_harmonics,
                        azimuth=observation.get_azimuth(),
                        hwp_angles=observation.get_hwp_angles(),
                        n_dets=observation.n_detectors,
                        dtype=config.dtype,
                        scan_mask=observation.get_right_scan_mask(),
                    )
                )
            else:
                blocks['azhwp_synchronous'] = (
                    templates.AzimuthHWPSynchronousTemplateOperator.create(
                        n_polynomials=azhwpss.n_polynomials,
                        n_harmonics=azhwpss.n_harmonics,
                        azimuth=observation.get_azimuth(),
                        hwp_angles=observation.get_hwp_angles(),
                        n_dets=observation.n_detectors,
                        dtype=config.dtype,
                    )
                )
        if binazhwpss := config.templates.binazhwp_synchronous:
            blocks['binazhwp_synchronous'] = (
                templates.BinAzimuthHWPSynchronousTemplateOperator.create(
                    n_azimuth_bins=binazhwpss.n_azimuth_bins,
                    n_harmonics=binazhwpss.n_harmonics,
                    interpolate_azimuth=binazhwpss.interpolate_azimuth,
                    smooth_interpolation=binazhwpss.smooth_interpolation,
                    azimuth=observation.get_azimuth(),
                    hwp_angles=observation.get_hwp_angles(),
                    n_dets=observation.n_detectors,
                    dtype=config.dtype,
                )
            )
        if ground := config.templates.ground:
            self._ground_landscape = templates.GroundTemplateOperator.get_landscape(
                azimuth_resolution=ground.azimuth_resolution,
                elevation_resolution=ground.elevation_resolution,
                boresight_azimuth=observation.get_azimuth(),
                boresight_elevation=observation.get_elevation(),
                detector_quaternions=observation.get_detector_quaternions(),
                stokes='IQU',
                dtype=config.dtype,
            )
            ground_op = templates.GroundTemplateOperator.create(
                azimuth_resolution=ground.azimuth_resolution,
                elevation_resolution=ground.elevation_resolution,
                boresight_azimuth=observation.get_azimuth(),
                boresight_elevation=observation.get_elevation(),
                boresight_rotation=jnp.zeros_like(observation.get_azimuth()),
                detector_quaternions=observation.get_detector_quaternions(),
                hwp_angles=observation.get_hwp_angles(),
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
        data = observation.get_tods().astype(config.dtype)
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
        if config.noise.fit_from_data:
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
        data = observation.get_tods().astype(config.dtype)
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
            correlation_length=config.noise.correlation_length,
        )
        noise = noise_model.operator(
            data_struct,
            sample_rate=observation.sample_rate,
            correlation_length=config.noise.correlation_length,
        )
        logger_info('Created noise and inverse noise covariance operators')

        # Approximate system matrix with diagonal noise covariance and full map pixels
        if not isinstance(inv_noise, SymmetricBandToeplitzOperator):
            raise NotImplementedError

        diag_inv_noise = DiagonalOperator(inv_noise.band_values[..., [0]], in_structure=data_struct)
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
                norm_sys = jnp.abs(
                    jax.jit(lambda x: tmpl_sys(x))(furax.tree.ones_like(tmpl_op.in_structure))
                )
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
            nested_solver = lx.CG(
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

        solver = lx.CG(**asdict(config.solver))
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

        # Format output and compute auxilary data
        final_map = np.array([result_map.i, result_map.q, result_map.u])

        output = {'map': final_map, 'weights': weights, 'weights_uncut': blocks}
        if isinstance(landscape, WCSLandscape):
            output['wcs'] = landscape.to_wcs()
        elif isinstance(landscape, AstropyWCSLandscape):
            output['wcs'] = landscape.wcs
        if config.noise.fit_from_data:
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
        data = observation.get_tods().astype(config.dtype)
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

        solver = lx.CG(**asdict(config.solver))
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

        # Format output and compute auxilary data
        final_map = np.array([result_map.i, result_map.q, result_map.u])

        output = {'map': final_map, 'weights': blocks}
        for key in tmpl_ampl.keys():
            output[f'template_{key}'] = tmpl_ampl[key]
        if isinstance(landscape, WCSLandscape):
            output['wcs'] = landscape.to_wcs()
        elif isinstance(landscape, AstropyWCSLandscape):
            output['wcs'] = landscape.wcs
        if config.noise.fit_from_data:
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
        data = observation.get_tods().astype(config.dtype)
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

        solver = lx.CG(**asdict(self.config.solver))
        spd = OperatorTag.POSITIVE_SEMIDEFINITE
        lx_system = as_lineax_operator(lhs, spd)
        lx_precond = as_lineax_operator(preconditioner.reduce(), spd)
        logger_info('Completed setting up the solver')

        # Run mapmaking
        rhs = rhs_op(data)
        y0 = preconditioner(rhs)
        solution = lx.linear_solve(
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

        # Format output and compute auxilary data
        final_map = np.array([result_map.q, result_map.u])

        output = {'map': final_map, 'weights': blocks}
        if isinstance(landscape, AstropyWCSLandscape):
            output['wcs'] = landscape.wcs
        if config.noise.fit_from_data:
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
