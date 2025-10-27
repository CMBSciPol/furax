from pathlib import Path

import jax
from jax import Array
from jaxtyping import PyTree
from sotodlib.core import AxisManager

from furax.mapmaking import GroundObservationReader

from .observation import SOTODLibObservation


@jax.tree_util.register_static
class SOTODLibReader(GroundObservationReader):
    """Jittable reader for SOTODlib observations.
    See GroundObservationReader for details.
    """

    def __init__(
        self, filenames: list[Path | str], data_field_names: list[str] | None = None
    ) -> None:
        """Initializes the SOTODLib reader with a list of filenames.

        Args:
            filenames: A list of filenames. Each filename can be a string or a Path object.
            data_field_names: A list of data fields to load. If None, read all available data fields
                available for SOTODLib observations.
        """
        super().__init__(filenames=filenames, data_field_names=data_field_names)

    def _read_structure_impure(
        self, path: Path, data_field_names: list[str]
    ) -> PyTree[jax.ShapeDtypeStruct]:
        filename = path.as_posix()

        # Load minimal amount to find data shape
        sotodlib_obs = AxisManager.load(filename, fields=['dets', 'samps'])
        n_detectors = sotodlib_obs.dets.count
        n_samples = sotodlib_obs.samps.count

        field_structure = GroundObservationReader.data_field_structures(
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
        if 'boresight_quaternions' in data_field_names:
            sub_fields.append('boresight')
            if 'timestamps' not in sub_fields:
                sub_fields.append('timestamps')
        if 'detector_quaternions' in data_field_names:  # the detector quaternions.
            sub_fields.append('focal_plane')

        sotodlib_obs = AxisManager.load(filename, fields=sub_fields)
        obs = SOTODLibObservation(sotodlib_obs)

        field_reader = GroundObservationReader.data_field_readers()
        return {field: field_reader[field](obs) for field in data_field_names}
