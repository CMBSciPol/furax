from pathlib import Path

import jax
from jax import Array
from jax.tree_util import register_static
from jaxtyping import PyTree
from sotodlib.core import AxisManager

from furax.mapmaking import AbstractGroundObservationReader

from .observation import SOTODLibObservation


@register_static
class SOTODLibReader(AbstractGroundObservationReader):
    """Class for handling a set of sotodlib ground observations.

    See AbstractGroundObservationReader for details.

    Usage:
        >>> reader = SOTODLibReader(['obs1.h5', 'obs2.h5'])
        >>> data1, padding1 = reader.read(0)
        >>> data2, padding2 = reader.read(1)
    """

    def _read_structure_impure(
        self, path: Path, data_field_names: list[str]
    ) -> PyTree[jax.ShapeDtypeStruct]:
        filename = path.as_posix()

        # Load minimal amount to find data shape
        sotodlib_obs = AxisManager.load(filename, fields=['dets', 'samps'])
        n_detectors = sotodlib_obs.dets.count
        n_samples = sotodlib_obs.samps.count

        field_structure = AbstractGroundObservationReader._get_data_field_structures_for(
            n_detectors=n_detectors, n_samples=n_samples
        )
        return {field: field_structure[field] for field in data_field_names}

    def _read_data_impure(self, path: Path, data_field_names: list[str]) -> PyTree[Array]:
        filename = path.as_posix()

        sub_fields = []
        if 'sample_data' in data_field_names:
            sub_fields.append('signal')
        if 'valid_sample_masks' in data_field_names:
            sub_fields.append('flags.glitch_flags')
        if 'valid_scanning_masks' in data_field_names:
            sub_fields.append('preprocess.turnaround_flags')
        if 'timestamps' in data_field_names:
            sub_fields.append('timestamps')
        if 'hwp_angles' in data_field_names:
            sub_fields.append('hwp_angle')
        if 'boresight_quaternions' in data_field_names:
            sub_fields.append('boresight')
            if 'timestamps' not in sub_fields:
                sub_fields.append('timestamps')
        if 'detector_quaternions' in data_field_names:  # the detector quaternions.
            sub_fields.append('focal_plane')

        sotodlib_obs = AxisManager.load(filename, fields=sub_fields)
        obs = SOTODLibObservation(sotodlib_obs)

        field_reader = AbstractGroundObservationReader._get_data_field_readers()
        return {field: field_reader[field](obs) for field in data_field_names}

    def update_data_field_names(self, data_field_names: list[str]) -> 'SOTODLibReader':
        return SOTODLibReader(*self.args, data_field_names=data_field_names)
