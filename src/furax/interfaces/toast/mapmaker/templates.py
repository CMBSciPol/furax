from typing import Any

import equinox
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int, PyTree
from numpy.typing import NDArray

from furax import AbstractLinearOperator

from .interface import ObservationData


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
        dtype: jnp.dtype | str = np.dtype(float),  # type: ignore[type-arg]
    ) -> None:
        self.n_params = n_params
        self.n_dets = n_dets
        self.n_samps = n_samps
        dtype = jnp.dtype(dtype)
        self._in_structure = jax.ShapeDtypeStruct((n_params,), dtype)
        self._out_structure = jax.ShapeDtypeStruct((n_dets, n_samps), dtype)

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return self._in_structure

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self._out_structure

    @classmethod
    def from_dict(
        cls, name: str, config: dict[str, Any], observation_data: ObservationData
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
            )

        elif name == 'scan_synchronous':
            min_poly_order: int = config.get('min_poly_order', 0)
            max_poly_order: int = config.get('max_poly_order', 0)  # type: ignore[no-redef]
            return ScanSynchronousTemplateOperator.create(
                min_poly_order=min_poly_order,
                max_poly_order=max_poly_order,
                azimuth=observation_data.get_azimuth(),
                n_dets=n_dets,
            )

        elif name == 'hwp_synchronous':
            n_harmonics: int = config.get('n_harmonics', 0)
            return HWPSynchronousTemplateOperator.create(
                n_harmonics=n_harmonics, hwp_angles=observation_data.get_hwp_angles(), n_dets=n_dets
            )

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

        raise NotImplementedError(f'Template {name} is not implemented')


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
        max_poly_order: int,
        blocks: PyTree[Float[Array, '...']],
        block_slice_indices: PyTree[int],
    ) -> None:
        # TODO: add checks
        super().__init__(n_params, n_dets, n_samps)
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
            )[0, :, :]

            res = jnp.concatenate(
                [
                    jnp.zeros((evals.shape[0], scan_start - block_start), dtype=evals.dtype),
                    evals,
                    jnp.zeros((evals.shape[0], block_end - scan_end), dtype=evals.dtype),
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
        min_poly_order: int,
        max_poly_order: int,
        templates: Float[Array, 'ord samp'],
    ):
        # TODO: add checks
        super().__init__(n_params, n_dets, n_samps)
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
        )[0, min_poly_order:, :]

        return cls(
            n_params=n_params,
            n_dets=n_dets,
            n_samps=n_samps,
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
        n_harmonics: int,
        templates: Float[Array, 'harm samp'],
    ):
        # TODO: add checks
        super().__init__(n_params, n_dets, n_samps)
        self.n_harmonics = n_harmonics
        self.templates = templates

    @classmethod
    def create(
        cls,
        n_harmonics: int,
        hwp_angles: Float[Array, ' samps'],
        n_dets: int,
    ) -> TemplateOperator:
        # 2 real components per harmonics
        n_samps = len(hwp_angles)
        n_params: int = n_dets * 2 * n_harmonics

        harmonics = jnp.arange(1, n_harmonics + 1)
        sines = jnp.sin(harmonics[:, None] * hwp_angles[None, :])
        cosines = jnp.cos(harmonics[:, None] * hwp_angles[None, :])

        templates = jnp.concatenate([sines, cosines], axis=0)

        return cls(
            n_params=n_params,
            n_dets=n_dets,
            n_samps=n_samps,
            n_harmonics=n_harmonics,
            templates=templates,
        )

    def mv(self, x: Float[Array, ' a']) -> Float[Array, '...']:
        params = x.reshape(self.n_dets, 2 * self.n_harmonics)

        # data[det,samp] = parameter[det,harm] * template[harm,samp]
        return params @ self.templates

    def transpose(self) -> AbstractLinearOperator:
        return HWPSynchronousTemplateTransposeOperator(self)


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
        n_modes: int,
        templates: Float[Array, 'mode samp'],
    ):
        # TODO: add checks
        super().__init__(n_params, n_dets, n_samps)
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
        templates = W.T @ tods

        return cls(
            n_params=n_params,
            n_dets=n_dets,
            n_samps=n_samps,
            n_modes=n_modes,
            templates=templates,
        )

    def mv(self, x: Float[Array, ' a']) -> Float[Array, '...']:
        params = x.reshape(self.n_dets, self.n_modes)

        # data[det,samp] = parameter[det,mode] * template[mode,samp]
        return params @ self.templates

    def transpose(self) -> AbstractLinearOperator:
        return CommonModeTemplateTransposeOperator(self)


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
