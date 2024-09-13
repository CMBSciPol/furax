import time
from pathlib import Path
from typing import TypeAlias

import healpy as hp
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import traitlets
from jaxtyping import PyTree
from toast.observation import default_values as defaults
from toast.ops.operator import Operator as ToastOperator
from toast.timing import Timer, function_timer
from toast.traits import Bool, Float, Int, Unicode, trait_docs
from toast.utils import Logger

from furax import Config
from furax.landscapes import HealpixLandscape, StokesIQUPyTree
from furax.operators import AbstractLinearOperator, DiagonalOperator, IdentityOperator
from furax.operators.hwp import HWPOperator
from furax.operators.polarizers import LinearPolarizerOperator
from furax.operators.projections import SamplingOperator
from furax.operators.qu_rotations import QURotationOperator
from furax.operators.toeplitz import SymmetricBandToeplitzOperator

from .interface import ObservationData
from .utils import estimate_psd, interpolate_psd, next_fast_fft_size, psd_to_invntt

ObservationKeysDict: TypeAlias = dict[str, list[str]]


@trait_docs
class MapMaker(ToastOperator):  # type: ignore[misc]
    """Operator which makes maps with the furax tools."""

    # Class traits
    API = Int(0, help='Internal interface version for this operator')

    # General configuration
    binned = Bool(False, help='Make a binned map')
    lagmax = Int(1_000, help='Maximum lag of the correlation function')
    nside = Int(64, help='HEALPix nside parameter for the output maps')
    output_dir = Unicode('.', help='Write output data products to this directory')
    stokes = Unicode('IQU', help='Stokes parameters to reconstruct')

    # Solver configuration
    atol = Float(1e-6, help='Absolute tolerance for terminating solve')
    rtol = Float(1e-6, help='Relative tolerance for terminating solve')
    max_steps = Int(500, help='Maximum number of iterations to run the solver for')

    # TOAST buffer names
    det_data = Unicode(defaults.det_data, help='Observation detdata key for the timestream data')
    hwp_angle = Unicode(None, allow_none=True, help='Observation shared key for HWP angle')
    noise_model = Unicode(
        None, allow_none=True, help='Observation key containing a noise model to use'
    )
    pixels = Unicode(defaults.pixels, help='Observation detdata key for pixel indices')
    quats = Unicode(defaults.quats, help='Observation detdata key for detector quaternions')

    # Flagging and masking
    det_mask = Int(defaults.det_mask_nonscience, help='Bit mask value for per-detector flagging')

    # TODO support this?
    view = Unicode(None, allow_none=True, help='Use this view of the data in all observations')

    @traitlets.validate('stokes')
    def _check_stokes(self, proposal):  # type: ignore[no-untyped-def]
        value = proposal['value']
        if value != 'IQU':
            not_iqu_error_msg = f'Only IQU map-making is supported for now, got {value!r}'
            raise traitlets.TraitError(not_iqu_error_msg)
        return value

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self._cached = False
        self._logprefix = 'Furax MapMaker:'

    def clear(self) -> None:
        """Delete the underlying memory."""
        for attr in [
            '_tods',
            '_pixels',
            '_quats',
            '_det_angles',
            '_gamma',
            '_hwp_angles',
        ]:
            if hasattr(self, attr):
                delattr(self, attr)

    def __del__(self) -> None:
        self.clear()

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):  # type: ignore[no-untyped-def]
        log = Logger.get()
        timer = Timer()
        timer.start()

        def log_info(msg: str) -> None:
            log.info_rank(
                f'{self._logprefix} {msg} in',
                comm=data.comm.comm_world,
                timer=timer,
            )

        if len(data.obs) == 0:
            raise RuntimeError('Every supplied data object must contain at least one observation.')

        if len(data.obs) > 1:
            raise RuntimeWarning(
                'Only one observation is supported for now. The first observation will be used.'
            )

        # Initialize the internal data interface
        self._data = ObservationData(
            observation=data.obs[0],
            det_selection=detectors,
            det_mask=self.det_mask,
            det_data=self.det_data,
            pixels=self.pixels,
            quats=self.quats,
            hwp_angle=self.hwp_angle,
            noise_model=self.noise_model,
        )

        # Check that we have at least one valid detector
        dets = self._data.dets
        ndet = len(dets)
        if ndet == 0:
            return

        # Stage data
        has_hwp = self.hwp_angle is not None
        self._stage(has_hwp)
        log_info('Staged data')

        # Build the acquisition operator
        h = self._get_acquisition(has_hwp)
        tod_structure = h.out_structure()
        log_info('Built acquisition')

        # Build the inverse noise covariance matrix
        invntt = self._get_invntt(tod_structure)
        log_info('Built invntt')

        # preconditioner
        # TODO replace by block Jacobi when available
        coverage = h.T(jnp.ones(tod_structure.shape, tod_structure.dtype))
        m = DiagonalOperator(
            StokesIQUPyTree(
                i=coverage.i,
                q=coverage.i,
                u=coverage.i,
            )
        ).inverse()

        # solving
        solver = lx.CG(rtol=self.rtol, atol=self.atol, max_steps=self.max_steps)
        solver_options = {
            'preconditioner': lx.TaggedLinearOperator(m, lx.positive_semidefinite_tag)
        }
        with Config(solver=solver, solver_options=solver_options):
            A = (h.T @ invntt @ h).I @ h.T @ invntt

        @jax.jit
        def process(tod):  # type: ignore[no-untyped-def]
            return A.reduce()(tod)

        time0 = time.perf_counter()
        estimate = process(self._tods)
        estimate.i.block_until_ready()
        print(f'JIT 1 .i: {time.perf_counter() - time0}')

        time0 = time.perf_counter()
        estimate = process(self._tods)
        estimate.i.block_until_ready()
        print(f'JIT 2 .i: {time.perf_counter() - time0}')

        log_info('Mapped the data')

        # save the result
        out_dir = Path(self.output_dir)
        out_dir.mkdir(exist_ok=True)
        fname = out_dir / 'map.fits'
        map_estimate = np.array([estimate.i, estimate.q, estimate.u])
        hp.write_map(str(fname), map_estimate, nest=True, overwrite=True)
        log_info('Saved outputs')

        return

    def _stage(self, hwp: bool) -> None:
        self._tods = self._data.get_tods()
        self._pixels = self._data.get_pixels()
        self._det_angles = self._data.get_det_angles()
        if not hwp:
            return
        self._hwp_angles = self._data.get_hwp_angles()
        self._gamma = self._data.get_offsets()

    def _get_invntt(self, structure: PyTree[jax.ShapeDtypeStruct]) -> AbstractLinearOperator:
        if self.binned:
            # we are making a binned map
            return IdentityOperator(structure)

        if self.lagmax > self._data.samples:
            raise RuntimeError(
                'Maximum correlation length should be less than the number of samples'
            )

        fft_size = next_fast_fft_size(self._data.samples)
        if self.noise_model is None:
            # estimate the noise covariance from the data
            psd = estimate_psd(self._tods, fft_size=fft_size, rate=self._data.sample_rate)
        else:
            # use an existing Noise model
            freq, psd = self._data.get_psd_model()
            psd = interpolate_psd(freq, psd, fft_size=fft_size, rate=self._data.sample_rate)
        invntt = psd_to_invntt(psd, self.lagmax)
        return SymmetricBandToeplitzOperator(invntt, structure)

    def _get_acquisition(self, has_hwp: bool) -> AbstractLinearOperator:
        self._landscape = HealpixLandscape(self.nside, self.stokes)
        sampling = SamplingOperator(landscape=self._landscape, indices=self._pixels)
        meta = {'shape': self._tods.shape, 'stokes': self.stokes}
        if has_hwp:
            polarizer = LinearPolarizerOperator.create(**meta, angles=self._gamma[:, None])
            hwp = HWPOperator.create(**meta, angles=self._hwp_angles[None, :])
            rotation = QURotationOperator.create(
                **meta, angles=self._det_angles - self._gamma[:, None]
            )
            acquisition = polarizer @ hwp @ rotation @ sampling
            reduced_acquisition = acquisition.reduce()
            return reduced_acquisition
        else:
            polarizer = LinearPolarizerOperator.create(**meta, angles=self._det_angles)
            # no need for reduction here
            return polarizer @ sampling

    def _finalize(self, data, **kwargs):  # type: ignore[no-untyped-def]
        self.clear()

    def _requires(self) -> ObservationKeysDict:
        req = {
            'meta': [],
            'shared': [],
            'detdata': [self.det_data, self.pixels, self.quats],
            'intervals': [],
        }
        for trait_name, destination in zip(
            ['noise_model', 'hwp_angle', 'view'], ['meta', 'shared', 'intervals']
        ):
            if (trait := getattr(self, trait_name)) is not None:
                req[destination].append(trait)
        return req

    def _provides(self) -> ObservationKeysDict:
        return dict()
