from abc import abstractmethod
from typing import Any

import equinox
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jaxtyping import DTypeLike, Float, Inexact, Int, PyTree
from numpy.typing import NDArray

from furax import AbstractLinearOperator
from furax.math import quaternion
from furax.obs import HWPOperator, LinearPolarizerOperator
from furax.obs.landscapes import HorizonLandscape
from furax.obs.stokes import Stokes, ValidStokesType

from . import GroundObservationData
from .pointing import PointingOperator


class TemplateOperator(AbstractLinearOperator):
    """Operator for time-ordered-data templates used for mapmaking.
    The input and output structures are fixed.
    """

    n_params: int = equinox.field(static=True)  # Number of template parameters
    n_dets: int = equinox.field(static=True)  # Number of detecotrs
    n_samps: int = equinox.field(static=True)  # Number of samples

    _in_structure: jax.ShapeDtypeStruct = equinox.field(static=True)
    _out_structure: jax.ShapeDtypeStruct = equinox.field(static=True)

    def __init__(
        self,
        n_params: int,
        n_dets: int,  # Number of detectors
        n_samps: int,  # Number of samples
        dtype: DTypeLike,
    ) -> None:
        self.n_params = n_params
        self.n_dets = n_dets
        self.n_samps = n_samps
        self._in_structure = jax.ShapeDtypeStruct((n_params,), dtype)
        self._out_structure = jax.ShapeDtypeStruct((n_dets, n_samps), dtype)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return self._in_structure

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self._out_structure

    @classmethod
    def from_dict(
        cls, name: str, config: dict[str, Any], observation_data: GroundObservationData
    ) -> AbstractLinearOperator:
        """Create and return a template operator corresponding to the
        name and configuration provided.
        """
        n_dets = len(observation_data.dets)

        if name == 'polynomial':
            max_poly_order: int = config.get('max_poly_order', 0)
            return PolynomialTemplateOperator.create(
                max_poly_order=max_poly_order,
                intervals=observation_data.get_scanning_intervals(),
                times=observation_data.get_elapsed_time(),
                n_dets=n_dets,
                dtype=config.get('dtype', jnp.float64),
            )

        elif name == 'scan_synchronous':
            min_poly_order: int = config.get('min_poly_order', 0)
            max_poly_order: int = config.get('max_poly_order', 0)  # type: ignore[no-redef]
            return ScanSynchronousTemplateOperator.create(
                min_poly_order=min_poly_order,
                max_poly_order=max_poly_order,
                azimuth=observation_data.get_azimuth(),
                n_dets=n_dets,
                dtype=config.get('dtype', jnp.float64),
            )

        elif name == 'hwp_synchronous':
            n_harmonics: int = config.get('n_harmonics', 0)
            return HWPSynchronousTemplateOperator.create(
                n_harmonics=n_harmonics,
                hwp_angles=observation_data.get_hwp_angles(),
                n_dets=n_dets,
                dtype=config.get('dtype', jnp.float64),
            )

        """
        elif name == 'common_mode':
            # Assumes that the cross power spectral density is precomputed
            # and stored as '_cross_psd'
            assert observation_data._cross_psd is not None
            freq, csd = observation_data._cross_psd
            freq_threshold: float = config.get('freq_threshold', 0.0)
            n_modes: int = config.get('n_modes', 0)

            return CommonModeTemplateOperator.create(
                freq_threshold=freq_threshold,
                n_modes=n_modes,
                freq=freq,
                csd=csd,
                tods=observation_data.get_tods(),
            )
        """

        raise NotImplementedError(f'Template {name} is not implemented')

    @abstractmethod
    def compute_auxiliary_data(self, template_amplitude: Float[Array, '...']) -> dict[str, Any]:
        """Compute optional data to be computed at the end for the best-fit template amplitudes."""
        ...


class PolynomialTemplateOperator(TemplateOperator):
    """Operator for polynomial trends up to a certain order.
    The template consists of blocks which spans scanning intervals,
    during which the polynomial coefficients are fixed for each detector.

    Stores legendre polynomial evaluations in 'blocks', a list of matrices
    with varying sizes, that addes up to (n_samps, max_poly_order+1)
    """

    max_poly_order: int = equinox.field(static=True)
    n_intervals: int = equinox.field(static=True)
    blocks: PyTree[Float[Array, '...']] = equinox.field(static=True)
    block_slice_indices: PyTree[int] = equinox.field(static=True)

    def __init__(
        self,
        n_params: int,
        n_dets: int,
        n_samps: int,
        dtype: DTypeLike,
        max_poly_order: int,
        blocks: PyTree[Float[Array, '...']],
        block_slice_indices: PyTree[int],
    ) -> None:
        # TODO: add checks
        super().__init__(n_params, n_dets, n_samps, dtype)
        self.max_poly_order = max_poly_order
        self.n_intervals = len(blocks)
        self.blocks = blocks
        self.block_slice_indices = block_slice_indices

    @classmethod
    def create(
        cls,
        max_poly_order: int,
        intervals: Int[Array, 'a 2'] | NDArray[np.int_],
        times: Float[Array, ' samps'],
        n_dets: int,
        dtype: DTypeLike,
    ) -> TemplateOperator:
        n_intervals = intervals.shape[0]
        n_params = n_intervals * n_dets * (max_poly_order + 1)
        n_samps = len(times)

        # Set blocks' starting and ending indices, so that
        # their union covers the entire sample
        block_starts = jnp.concatenate([jnp.array([0], dtype=int), intervals[1:, 0]])
        block_ends = jnp.concatenate([intervals[1:, 0], jnp.array([n_samps], dtype=int)])

        # This has to be a static list of integers
        block_slice_indices = [ind for ind in intervals[1:, 0]]

        def eval_legs_block(block_start, block_end, scan_start, scan_end) -> Float[Array, '...']:  # type: ignore[no-untyped-def]
            # Evaluates the Legendre polynomials of given orders on a block,
            # setting all values outside the scan range to zero.
            t = times[scan_start:scan_end]
            x = -1.0 + 2 * (t - t[0]) / (t[-1] - t[0])

            # This funtion computes a lot more than what we need,
            # but shouldn't be too expensive for low polynomial orders
            evals = jax.scipy.special.lpmn_values(
                max_poly_order, max_poly_order, x, is_normalized=False
            )[0, :, :].astype(dtype)

            res = jnp.concatenate(
                [
                    jnp.zeros((evals.shape[0], scan_start - block_start), dtype=dtype),
                    evals,
                    jnp.zeros((evals.shape[0], block_end - scan_end), dtype=dtype),
                ],
                axis=1,
            )
            return res

        # Unfortunately no vectorisation here since the block sizes vary dynamically
        blocks = [
            eval_legs_block(block_starts[i], block_ends[i], intervals[i, 0], intervals[i, 1])
            for i in range(n_intervals)
        ]

        return cls(
            n_params=n_params,
            n_dets=n_dets,
            n_samps=n_samps,
            dtype=dtype,
            max_poly_order=max_poly_order,
            blocks=blocks,
            block_slice_indices=block_slice_indices,
        )

    def mv(self, x: Float[Array, ' a']) -> Array:
        # At each block,
        # data[det,samp] = parameter[det,ord] * template[ord,samp]

        params = x.reshape(self.n_intervals, self.n_dets, self.max_poly_order + 1)

        block_mvs = [
            jnp.einsum('ij,jk->ik', param, block) for param, block in zip(params, self.blocks)
        ]

        # Again, we can't run below due to varying block sizes
        # block_mvs = jnp.einsum('ijk,ikl->ijl', params, self.blocks)

        return jnp.concatenate(block_mvs, axis=1)

    def transpose(self) -> AbstractLinearOperator:
        return PolynomialTemplateTransposeOperator(self)

    def compute_auxiliary_data(self, template_amplitude: Float[Array, '...']) -> dict[str, Any]:
        """Compute optional data to be computed at the end for the best-fit template amplitudes."""
        return {}


class PolynomialTemplateTransposeOperator(AbstractLinearOperator):
    operator: PolynomialTemplateOperator

    def mv(self, x: Float[Array, 'a b']) -> Float[Array, '...']:
        # At each block,
        # parameter[det,ord] = data[det,samp] * template[ord,samp]

        # This slicing works as block_slice_indices is a static list
        datas = jnp.array_split(x, self.operator.block_slice_indices, axis=1)
        block_mvs = [
            jnp.einsum('ij,kj->ik', data, block) for data, block in zip(datas, self.operator.blocks)
        ]

        return jnp.concatenate([b.ravel() for b in block_mvs])

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure()


class ScanSynchronousTemplateOperator(TemplateOperator):
    """Operator for scan-synchronous signal templates.
    The template is a legendre polynomial of the scan azimuth,
    with individual polynomial coefficients for each detector.
    """

    min_poly_order: int = equinox.field(static=True)
    max_poly_order: int = equinox.field(static=True)
    templates: Float[Array, 'ord samp'] = equinox.field(static=True)

    def __init__(
        self,
        n_params: int,
        n_dets: int,
        n_samps: int,
        dtype: DTypeLike,
        min_poly_order: int,
        max_poly_order: int,
        templates: Float[Array, 'ord samp'],
    ):
        # TODO: add checks
        super().__init__(n_params, n_dets, n_samps, dtype)
        self.min_poly_order = min_poly_order
        self.max_poly_order = max_poly_order
        self.templates = templates

    @classmethod
    def create(
        cls,
        min_poly_order: int,
        max_poly_order: int,
        azimuth: Float[Array, ' n_samps'],
        n_dets: int,
        dtype: DTypeLike,
    ) -> TemplateOperator:
        n_samps = len(azimuth)
        n_params: int = n_dets * (max_poly_order - min_poly_order + 1)

        max_az = jnp.max(azimuth)
        min_az = jnp.min(azimuth)
        x = -1.0 + 2.0 * (azimuth - min_az) / (max_az - min_az)

        # This funtion computes a lot more than what we need,
        # but shouldn't be too expensive for low polynomial orders
        templates = jax.scipy.special.lpmn_values(
            max_poly_order, max_poly_order, x, is_normalized=False
        )[0, min_poly_order:, :].astype(dtype)

        return cls(
            n_params=n_params,
            n_dets=n_dets,
            n_samps=n_samps,
            dtype=dtype,
            min_poly_order=min_poly_order,
            max_poly_order=max_poly_order,
            templates=templates,
        )

    def mv(self, x: Float[Array, ' a']) -> Float[Array, '...']:
        n_poly: int = self.max_poly_order - self.min_poly_order + 1
        params = x.reshape(self.n_dets, n_poly)

        # data[det,samp] = parameter[det,ord] * template[ord,samp]
        return params @ self.templates

    def transpose(self) -> AbstractLinearOperator:
        return ScanSynchronousTemplateTransposeOperator(self)

    def compute_auxiliary_data(self, template_amplitude: Float[Array, '...']) -> dict[str, Any]:
        """Compute optional data to be computed at the end for the best-fit template amplitudes."""
        return {}


class ScanSynchronousTemplateTransposeOperator(AbstractLinearOperator):
    operator: ScanSynchronousTemplateOperator

    def mv(self, x: Float[Array, 'det samp']) -> Float[Array, 'det ord']:
        # parameter[det,ord] = data[det,samp] * template[ord,samp]

        params = jnp.einsum('ij,kj->ik', x, self.operator.templates)
        return params.ravel()

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure()


class HWPSynchronousTemplateOperator(TemplateOperator):
    """Operator for HWP-synchronous signal templates.
    The template consists of harmonics of form exp[i*k*phi],
    where k = 1,..,n_harmonics, and phi is the HWP angle.
    The harmonic coefficients are unique per detector and
    are assumed to be constant throughout the scan.
    """

    n_harmonics: int = equinox.field(static=True)
    templates: Float[Array, 'harm samp'] = equinox.field(static=True)

    def __init__(
        self,
        n_params: int,
        n_dets: int,
        n_samps: int,
        dtype: DTypeLike,
        n_harmonics: int,
        templates: Float[Array, 'harm samp'],
    ):
        # TODO: add checks
        super().__init__(n_params, n_dets, n_samps, dtype)
        self.n_harmonics = n_harmonics
        self.templates = templates

    @classmethod
    def create(
        cls,
        n_harmonics: int,
        hwp_angles: Float[Array, ' samps'],
        n_dets: int,
        dtype: DTypeLike,
    ) -> TemplateOperator:
        # 2 real components per harmonics
        n_samps = len(hwp_angles)
        n_params: int = n_dets * 2 * n_harmonics

        harmonics = jnp.arange(1, n_harmonics + 1)
        sines = jnp.sin(harmonics[:, None] * hwp_angles[None, :])
        cosines = jnp.cos(harmonics[:, None] * hwp_angles[None, :])

        templates = jnp.concatenate([sines, cosines], axis=0).astype(dtype)

        return cls(
            n_params=n_params,
            n_dets=n_dets,
            n_samps=n_samps,
            dtype=dtype,
            n_harmonics=n_harmonics,
            templates=templates,
        )

    def mv(self, x: Float[Array, ' a']) -> Float[Array, '...']:
        params = x.reshape(self.n_dets, 2 * self.n_harmonics)

        # data[det,samp] = parameter[det,harm] * template[harm,samp]
        return params @ self.templates

    def transpose(self) -> AbstractLinearOperator:
        return HWPSynchronousTemplateTransposeOperator(self)

    def compute_auxiliary_data(self, template_amplitude: Float[Array, '...']) -> dict[str, Any]:
        """Compute optional data to be computed at the end for the best-fit template amplitudes."""
        return {}


class HWPSynchronousTemplateTransposeOperator(AbstractLinearOperator):
    operator: HWPSynchronousTemplateOperator

    def mv(self, x: Float[Array, 'det samp']) -> Float[Array, 'det harm']:
        # parameter[det,harm] = data[det,samp] * template[harm,samp]

        params = jnp.einsum('ij,kj->ik', x, self.operator.templates)
        return params.ravel()

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure()


class CommonModeTemplateOperator(TemplateOperator):
    """Operator for common modes templates obtained by, e.g., PCA.
    The template consists of harmonics of form exp[i*k*phi],
    where k = 1,..,n_harmonics, and phi is the HWP angle.
    The harmonic coefficients are unique per detector and
    are assumed to be constant throughout the scan.
    """

    n_modes: int = equinox.field(static=True)
    templates: Float[Array, 'mode samp'] = equinox.field(static=True)

    def __init__(
        self,
        n_params: int,
        n_dets: int,
        n_samps: int,
        dtype: DTypeLike,
        n_modes: int,
        templates: Float[Array, 'mode samp'],
    ):
        # TODO: add checks
        super().__init__(n_params, n_dets, n_samps, dtype)
        self.n_modes = n_modes
        self.templates = templates

    @classmethod
    def create(
        cls,
        freq_threshold: float,
        n_modes: int,
        freq: Float[Array, ' freq'],
        csd: Float[Array, 'det det freq'],
        tods: Float[Array, 'det samps'],
        dtype: DTypeLike,
    ) -> TemplateOperator:
        n_dets, n_samps = tods.shape
        n_params: int = n_dets * n_modes

        # Take low frequencies excluding zero
        f_slice = jnp.logical_and(freq < freq_threshold, freq > 0)
        low_pass_csd = jnp.sum(jnp.where(f_slice, csd, 0.0), axis=-1)

        # Eigen decomposition
        _, evecs = jnp.linalg.eigh(low_pass_csd)

        # Select eigenvectors with largest eigenvalues
        W = evecs[:, -n_modes:]

        # Compute the template
        templates = (W.T @ tods).astype(dtype)

        return cls(
            n_params=n_params,
            n_dets=n_dets,
            n_samps=n_samps,
            dtype=dtype,
            n_modes=n_modes,
            templates=templates,
        )

    def mv(self, x: Float[Array, ' a']) -> Float[Array, '...']:
        params = x.reshape(self.n_dets, self.n_modes)

        # data[det,samp] = parameter[det,mode] * template[mode,samp]
        return params @ self.templates

    def transpose(self) -> AbstractLinearOperator:
        return CommonModeTemplateTransposeOperator(self)

    def compute_auxiliary_data(self, template_amplitude: Float[Array, '...']) -> dict[str, Any]:
        """Compute optional data to be computed at the end for the best-fit template amplitudes."""
        return {}


class CommonModeTemplateTransposeOperator(AbstractLinearOperator):
    operator: CommonModeTemplateOperator

    def mv(self, x: Float[Array, 'det samp']) -> Float[Array, 'det mode']:
        # parameter[det,mode] = data[det,samp] * template[mode,samp]

        params = jnp.einsum('ij,kj->ik', x, self.operator.templates)
        return params.ravel()

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure()


class AzimuthHWPSynchronousTemplateOperator(TemplateOperator):
    """Operator for azimuth and HWP-synchronous signal templates.
    The template includes both azimuth and HWP-synchronous signals,
    as well as some azimuth-dependent HWP harmonics.
    Tx = sum_n P_n(az)(x_n + sum_m (y_nm cos(m phi) + z_nm sin(m phi))),
    where n = 0, ..., n_polynomials-1, m = 1,..,n_harmonics,
    az is the azimuth angle, and phi is the HWP angle.
    The coefficients are unique per detector.
    """

    n_polynomials: int = equinox.field(static=True)
    n_harmonics: int = equinox.field(static=True)
    poly_templates: Float[Array, 'poly samp']
    harm_templates: Float[Array, 'harm samp']
    min_azimuth: Float[Array, '']
    max_azimuth: Float[Array, '']

    def __init__(
        self,
        n_params: int,
        n_dets: int,
        n_samps: int,
        dtype: DTypeLike,
        n_polynomials: int,
        n_harmonics: int,
        poly_templates: Float[Array, 'poly samp'],
        harm_templates: Float[Array, 'harm samp'],
        min_azimuth: Float[Array, ''],
        max_azimuth: Float[Array, ''],
    ):
        # TODO: add checks
        super().__init__(n_params, n_dets, n_samps, dtype)
        self.n_polynomials = n_polynomials
        self.n_harmonics = n_harmonics
        self.poly_templates = poly_templates
        self.harm_templates = harm_templates
        self.min_azimuth = min_azimuth
        self.max_azimuth = max_azimuth

    @classmethod
    def create(
        cls,
        n_polynomials: int,
        n_harmonics: int,
        azimuth: Float[Array, ' samps'],
        hwp_angles: Float[Array, ' samps'],
        n_dets: int,
        dtype: DTypeLike,
        scan_mask: Float[Array, ' samps'] | None = None,
    ) -> TemplateOperator:
        n_samps: int = azimuth.size
        n_params: int = n_dets * n_polynomials * (1 + 2 * n_harmonics)

        # Create azimuth templates
        min_az = jnp.min(azimuth)
        max_az = jnp.max(azimuth)
        x = -1.0 + 2.0 * (azimuth - min_az) / (max_az - min_az)

        # This funtion computes (n_polynomials)-times more than what we need,
        # but this shouldn't be too expensive for low polynomial orders
        poly_templates = jax.scipy.special.lpmn_values(
            n_polynomials - 1, n_polynomials - 1, x, is_normalized=False
        )[0, :, :].astype(dtype)

        if scan_mask is not None:
            poly_templates = scan_mask[None, :] * poly_templates

        # Create harmonic templates
        harmonics = jnp.arange(1, n_harmonics + 1)
        ones = jnp.ones((1, n_samps), dtype=dtype)
        sines = jnp.sin(harmonics[:, None] * hwp_angles[None, :])
        cosines = jnp.cos(harmonics[:, None] * hwp_angles[None, :])
        harm_templates = jnp.concatenate([ones, sines, cosines], axis=0)

        return cls(
            n_params=n_params,
            n_dets=n_dets,
            n_samps=n_samps,
            dtype=dtype,
            n_polynomials=n_polynomials,
            n_harmonics=n_harmonics,
            poly_templates=poly_templates,
            harm_templates=harm_templates,
            min_azimuth=min_az,
            max_azimuth=max_az,
        )

    def mv(self, x: Float[Array, ' a']) -> Float[Array, '...']:
        # Params are ordered as [det,poly,[x, y/z[harm]]]
        # So for a single detector, the ordering goes
        # x_0, y_01, ... y_0M, z_01, ... z_0M, x_1, y_11, ..., z_(N-1)M
        x = x.reshape(self.n_dets, self.n_polynomials, 1 + 2 * self.n_harmonics)

        # data[det_samp] = poly_tmpl[poly,samp] * param[det,poly,harm] * harm_tmpl[harm,samp]
        return jnp.sum((x @ self.harm_templates) * self.poly_templates[None, :, :], axis=1)

    def transpose(self) -> AbstractLinearOperator:
        return AzimuthHWPSynchronousTemplateTransposeOperator(self)

    def compute_auxiliary_data(self, template_amplitude: Float[Array, '...']) -> dict[str, Any]:
        """Compute optional data to be computed at the end for the best-fit template amplitudes."""
        # Create azimuth grid to sample from
        max_az = self.max_azimuth
        min_az = self.min_azimuth
        N = 1000
        azimuth = jnp.linspace(min_az, max_az, N)
        x = -1.0 + 2.0 * (azimuth - min_az) / (max_az - min_az)

        # Compute polynomials for the grid
        poly_tmpl = jax.scipy.special.lpmn_values(
            self.n_polynomials - 1, self.n_polynomials - 1, x, is_normalized=False
        )[0, :, :].astype(template_amplitude.dtype)

        # Compute
        # data[det,harm,samp] = ampl[det,poly,harm] * poly_tmpl[poly,samp] (sum over poly)
        az_fit = jnp.einsum(
            'dph,ps->dhs',
            template_amplitude.reshape(self.n_dets, self.n_polynomials, 1 + 2 * self.n_harmonics),
            poly_tmpl,
        )
        return {'azimuth_grid': azimuth, 'fit_at_azimuth_grid': az_fit}


class AzimuthHWPSynchronousTemplateTransposeOperator(AbstractLinearOperator):
    operator: AzimuthHWPSynchronousTemplateOperator

    def mv(self, x: Float[Array, 'det samp']) -> Float[Array, ' param']:
        # param[det,poly,harm] = poly_tmpl[poly,samp] * harm_tmpl[harm,samp] * data[det,samp]
        """
        # Slower but more memory efficient
        return jnp.array([
                jnp.sum((self.operator.poly_templates[:,None,:]
                        * self.operator.harm_templates[None,:,:]
                        * x[idet,None,None,:]), axis=-1).ravel()
                        for idet in range(self.operator.n_dets)])
        """
        # Faster but less memory efficient
        return jnp.sum(
            (
                self.operator.poly_templates[None, :, None, :]
                * self.operator.harm_templates[None, None, :, :]
                * x[:, None, None, :]
            ),
            axis=-1,
        ).ravel()

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure()


class BinAzimuthHWPSynchronousTemplateOperator(TemplateOperator):
    """Operator for azimuth and HWP-synchronous signal templates.
    The template includes both azimuth and HWP-synchronous signals,
    as well as some azimuth-dependent HWP harmonics.
    Tx = sum_n B_n(az)(x_n + sum_m (y_nm cos(m phi) + z_nm sin(m phi))),
    where n = 0, ..., n_az_bins-1, m = 1,..,n_harmonics,
    az is the azimuth angle, and phi is the HWP angle.
    Unlike AzimuthHWPSynchronousTemplateOperator, azimuth are binned.
    The coefficients are unique per detector.
    """

    n_azimuth_bins: int = equinox.field(static=True)
    n_harmonics: int = equinox.field(static=True)
    bin_templates: Float[Array, 'bin samp']
    harm_templates: Float[Array, 'harm samp']

    def __init__(
        self,
        n_params: int,
        n_dets: int,
        n_samps: int,
        dtype: DTypeLike,
        n_azimuth_bins: int,
        n_harmonics: int,
        bin_templates: Float[Array, 'poly samp'],
        harm_templates: Float[Array, 'harm samp'],
    ):
        # TODO: add checks
        super().__init__(n_params, n_dets, n_samps, dtype)
        self.n_azimuth_bins = n_azimuth_bins
        self.n_harmonics = n_harmonics
        self.bin_templates = bin_templates
        self.harm_templates = harm_templates

    @classmethod
    def create(
        cls,
        n_azimuth_bins: int,
        n_harmonics: int,
        interpolate_azimuth: bool,
        smooth_interpolation: bool,
        azimuth: Float[Array, ' samps'],
        hwp_angles: Float[Array, ' samps'],
        n_dets: int,
        dtype: DTypeLike,
    ) -> TemplateOperator:
        n_samps: int = azimuth.size
        n_params: int = n_dets * n_azimuth_bins * (1 + 2 * n_harmonics)

        # Create azimuth templates
        max_az = jnp.max(azimuth)
        min_az = jnp.min(azimuth)
        max_az += 1e-8  # Allow max_az to be included in the last bin

        az_bin_edges = jnp.linspace(min_az, max_az, n_azimuth_bins + 1)
        bin_inds = jnp.digitize(azimuth, az_bin_edges[1:])

        if interpolate_azimuth:
            az_bin_centres = 0.5 * (az_bin_edges[:-1] + az_bin_edges[1:])
            delta_bin = (max_az - min_az) / n_azimuth_bins

            if smooth_interpolation:
                # Sin^2 interpolation
                bin_templates = (
                    jnp.sin(
                        (jnp.pi / 2)
                        * (
                            jnp.clip(
                                1 - jnp.abs(azimuth[None, :] - az_bin_centres[:, None]) / delta_bin,
                                min=0,
                            )
                        )
                    )
                    ** 2
                )
                bin_templates /= jnp.sum(bin_templates, axis=0)[None, :]
            else:
                # Linear interpolation
                bin_templates = (
                    jnp.clip(
                        1 - jnp.abs(azimuth[None, :] - az_bin_centres[:, None]) / delta_bin, min=0
                    )
                    .at[0, azimuth < az_bin_centres[0]]
                    .set(1.0)
                    .at[-1, azimuth > az_bin_centres[-1]]
                    .set(1.0)
                )
        else:
            # Binned weights (0 or 1)
            bin_templates = (
                jnp.zeros((n_azimuth_bins, n_samps), dtype=dtype)
                .at[bin_inds, jnp.arange(n_samps)]
                .set(1.0)
            )

        # Create harmonic templates
        harmonics = jnp.arange(1, n_harmonics + 1)
        ones = jnp.ones((1, n_samps), dtype=dtype)
        sines = jnp.sin(harmonics[:, None] * hwp_angles[None, :])
        cosines = jnp.cos(harmonics[:, None] * hwp_angles[None, :])
        harm_templates = jnp.concatenate([ones, sines, cosines], axis=0)

        return cls(
            n_params=n_params,
            n_dets=n_dets,
            n_samps=n_samps,
            dtype=dtype,
            n_azimuth_bins=n_azimuth_bins,
            n_harmonics=n_harmonics,
            bin_templates=bin_templates,
            harm_templates=harm_templates,
        )

    def mv(self, x: Float[Array, ' a']) -> Float[Array, '...']:
        # Params are ordered as [det,poly,[x, y/z[harm]]]
        # So for a single detector, the ordering goes
        # x_0, y_01, ... y_0M, z_01, ... z_0M, x_1, y_11, ..., z_(N-1)M
        x = x.reshape(self.n_dets, self.n_azimuth_bins, 1 + 2 * self.n_harmonics)

        # data[det_samp] = bin_tmpl[bin,samp] * param[det,bin,harm] * harm_tmpl[harm,samp]
        return jnp.sum((x @ self.harm_templates) * self.bin_templates[None, :, :], axis=1)

    def transpose(self) -> AbstractLinearOperator:
        return BinAzimuthHWPSynchronousTemplateTransposeOperator(self)

    def compute_auxiliary_data(self, template_amplitude: Float[Array, '...']) -> dict[str, Any]:
        """Compute optional data to be computed at the end for the best-fit template amplitudes."""
        return {}


class BinAzimuthHWPSynchronousTemplateTransposeOperator(AbstractLinearOperator):
    operator: BinAzimuthHWPSynchronousTemplateOperator

    def mv(self, x: Float[Array, 'det samp']) -> Float[Array, ' param']:
        # param[det,bin,harm] = bin_tmpl[bin,samp] * harm_tmpl[harm,samp] * data[det,samp]
        """
        # Slower but more memory efficient
        return jnp.array([
                jnp.sum((self.operator.bin_templates[:,None,:]
                        * self.operator.harm_templates[None,:,:]
                        * x[idet,None,None,:]), axis=-1).ravel()
                        for idet in range(self.operator.n_dets)])
        """
        # Faster but less memory efficient
        return jnp.sum(
            (
                self.operator.bin_templates[None, :, None, :]
                * self.operator.harm_templates[None, None, :, :]
                * x[:, None, None, :]
            ),
            axis=-1,
        ).ravel()

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure()


class GroundTemplateOperator(TemplateOperator):
    """Operator for ground signal templates. The template amplitudes
    form a two-dimensional (elevation, azimuth) IQU map of the ground
    observed that is shared across detectors for the observation range.
    This class only contains a factory method.
    All argument angles should be provided in radians.
    """

    @classmethod
    def create(
        cls,
        azimuth_resolution: float,
        elevation_resolution: float,
        boresight_azimuth: Float[Array, ' samps'],
        boresight_elevation: Float[Array, ' samps'],
        boresight_rotation: Float[Array, ' samps'],
        detector_quaternions: Float[Array, 'dets 4'],
        hwp_angles: Float[Array, ' samps'],
        stokes: ValidStokesType,
        dtype: DTypeLike,
        landscape: HorizonLandscape | None = None,
        chunk_size: int = 0,
    ) -> AbstractLinearOperator:
        # Compute landscape if not provided
        if landscape is None:
            horizon_landscape: HorizonLandscape = cls.get_landscape(
                azimuth_resolution=azimuth_resolution,
                elevation_resolution=elevation_resolution,
                boresight_azimuth=boresight_azimuth,
                boresight_elevation=boresight_elevation,
                detector_quaternions=detector_quaternions,
                stokes=stokes,
                dtype=dtype,
            )
        else:
            horizon_landscape = landscape

        # Azimuth increases in an opposite way to longitude
        boresight_quaternions = quaternion.from_lonlat_angles(
            -boresight_azimuth, boresight_elevation, boresight_rotation
        )
        _, _, det_gamma = quaternion.to_xieta_angles(detector_quaternions)

        n_dets = detector_quaternions.shape[0]
        n_samps = boresight_azimuth.size

        pointing = PointingOperator(
            landscape=horizon_landscape,
            qbore=boresight_quaternions,
            qdet=detector_quaternions,
            det_gamma=det_gamma,
            _in_structure=horizon_landscape.structure,
            _out_structure=Stokes.class_for(stokes).structure_for((n_dets, n_samps)),
            chunk_size=chunk_size,
        )

        polarizer = LinearPolarizerOperator.create(
            shape=(n_dets, n_samps),
            dtype=dtype,
            stokes=stokes,
            angles=det_gamma[:, None].astype(dtype),
        )

        if stokes == 'I':
            return polarizer @ pointing

        hwp = HWPOperator.create(
            shape=(n_dets, n_samps), dtype=dtype, stokes=stokes, angles=hwp_angles.astype(dtype)
        )

        return polarizer @ hwp @ pointing

    @classmethod
    def get_landscape(
        cls,
        azimuth_resolution: float,
        elevation_resolution: float,
        boresight_azimuth: Float[Array, ' samps'],
        boresight_elevation: Float[Array, ' samps'],
        detector_quaternions: Float[Array, 'dets 4'],
        stokes: ValidStokesType,
        dtype: DTypeLike,
    ) -> HorizonLandscape:
        # First, set up a grid of (az, el) pairs
        n_grid = 10
        az_grid = jnp.linspace(jnp.min(boresight_azimuth), jnp.max(boresight_azimuth), n_grid)
        el_grid = jnp.linspace(jnp.min(boresight_elevation), jnp.max(boresight_elevation), n_grid)
        az_mesh, el_mesh = jnp.meshgrid(az_grid, el_grid, indexing='ij')
        qbore_mesh = quaternion.from_lonlat_angles(
            -az_mesh, el_mesh, jnp.zeros_like(az_mesh)
        )  # (ndet,N_GRID,N_GRID,4)
        qfull_mesh = quaternion.qmul(
            qbore_mesh[None, :, :, :], detector_quaternions[:, None, None, :]
        )
        det_az_mesh, det_el_mesh, _ = quaternion.to_lonlat_angles(qfull_mesh)
        det_az_mesh = -det_az_mesh

        # Azimuth angle is first restricted to to [0,2pi),
        # and unwrapped along the elevation grid, azimuth grid, and detector axes in order
        det_az_mesh = jnp.unwrap(
            jnp.unwrap(jnp.unwrap(det_az_mesh % (2 * jnp.pi), axis=2), axis=1), axis=0
        )

        # Allow small margins
        az_min = jnp.min(det_az_mesh) - 1e-4
        az_max = jnp.max(det_az_mesh) + 1e-4
        el_min = jnp.min(det_el_mesh) - 1e-4
        el_max = jnp.max(det_el_mesh) + 1e-4

        n_alt = int(np.ceil((el_max - el_min) / elevation_resolution))
        n_az = int(np.ceil((az_max - az_min) / azimuth_resolution))

        landscape = HorizonLandscape(
            shape=(n_az, n_alt),
            altitude_limits=(el_min, el_max),
            azimuth_limits=(az_min, az_max),
            stokes=stokes,
            dtype=dtype,
        )

        return landscape

    def compute_auxiliary_data(self, template_amplitude: Float[Array, '...']) -> dict[str, Any]:
        """Compute optional data to be computed at the end for the best-fit template amplitudes."""
        return {}


class StokesIQUFlattenOperator(AbstractLinearOperator):
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        self._in_structure = in_structure

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> Inexact[Array, ' _b']:
        return jnp.concatenate([x.i, x.q, x.u])
