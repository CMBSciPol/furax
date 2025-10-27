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
        manager = AxisManager.load(filename)
        obs = SOTODLibObservation(manager)
        n_detectors = obs.n_detectors
        n_samples = obs.n_samples

        field_structure = GroundObservationReader.data_field_structures(
            n_detectors=n_detectors, n_samples=n_samples
        )
        return {field: field_structure[field] for field in data_field_names}

    def _read_data_impure(self, path: Path, data_field_names: list[str]) -> PyTree[Array]:
        filename = path.as_posix()
        manager = AxisManager.load(filename)
        obs = SOTODLibObservation(manager)

        field_reader = GroundObservationReader.data_field_readers()
        return {field: field_reader[field](obs) for field in data_field_names}
