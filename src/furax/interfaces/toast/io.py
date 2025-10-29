from pathlib import Path

import jax
import toast.ops.load_hdf5
from jax import Array
from jax.tree_util import register_static
from jaxtyping import PyTree
from toast import Data
from toast.observation import default_values as toast_defaults

from furax.mapmaking import AbstractGroundObservationReader

from .observation import ToastObservation


@register_static
class ToastReader(AbstractGroundObservationReader):
    """Class for handling a set of TOAST ground observations.

    See AbstractGroundObservationReader for details.

    Usage:
        >>> reader = ToastReader(['obs1.h5', 'obs2.h5'])
        >>> data1, padding1 = reader.read(0)
        >>> data2, padding2 = reader.read(1)
    """

    def _read_structure_impure(
        self, path: Path, data_field_names: list[str]
    ) -> PyTree[jax.ShapeDtypeStruct]:
        filename = path.as_posix()

        # Load minimal amount to find data shape
        toast_data = Data()
        toast_loader = toast.ops.load_hdf5.LoadHDF5(
            files=filename,
            detdata=[],
            shared=[toast_defaults.times],
            intervals=[''],
        )
        toast_loader.apply(toast_data)
        obs = ToastObservation(toast_data)
        n_detectors = obs.n_detectors
        n_samples = obs.n_samples

        field_structure = AbstractGroundObservationReader._get_data_field_structures_for(
            n_detectors=n_detectors, n_samples=n_samples
        )
        return {field: field_structure[field] for field in data_field_names}

    def _read_data_impure(self, path: Path, data_field_names: list[str]) -> PyTree[Array]:
        filename = path.as_posix()

        detdata = []
        if 'sample_data' in data_field_names:
            detdata.append(toast_defaults.det_data)
        if 'valid_sample_masks' in data_field_names:
            detdata.append(toast_defaults.det_flags)

        shared = [toast_defaults.times]  # Always need to load timestamps
        if 'hwp_angles' in data_field_names:
            shared.append(toast_defaults.hwp_angle)
        if 'boresight_quaternions' in data_field_names:
            shared.append(toast_defaults.boresight_radec)

        intervals = ['']  # Toast loads all intervals if the list is empty
        if 'valid_scanning_masks' in data_field_names:
            intervals.append(toast_defaults.scanning_interval)

        toast_data = Data()
        toast_loader = toast.ops.load_hdf5.LoadHDF5(
            files=filename,
            detdata=detdata,
            shared=shared,
            intervals=intervals,
        )
        toast_loader.apply(toast_data)
        obs = ToastObservation(toast_data)

        field_reader = AbstractGroundObservationReader._get_data_field_readers()
        return {field: field_reader[field](obs) for field in data_field_names}

    def update_data_field_names(self, data_field_names: list[str]) -> 'ToastReader':
        # re-create original args
        # this turns [(a1, b1), (a2, b2), (a3, b3)] -> ([a1, a2, a3], [b1, b2, b3])
        args = tuple(map(list, zip(*self.args)))
        return ToastReader(*args, data_field_names=data_field_names)
