from dataclasses import dataclass
from functools import cached_property

import jax.numpy as jnp
import numpy as np
import toast
from astropy import units as u
from jaxtyping import Array
from toast.observation import default_values as defaults

from .utils import get_local_meridian_angle


@dataclass
class ObservationData:
    observation: toast.Observation
    det_selection: list[str] | None = None
    det_mask: int = defaults.det_mask_nonscience

    # the names of the fields we need
    det_data: str = defaults.det_data
    pixels: str = defaults.pixels
    quats: str = defaults.quats
    hwp_angle: str | None = defaults.hwp_angle
    noise_model: str | None = defaults.noise_model

    @property
    def samples(self) -> int:
        return self.observation.n_local_samples  # type: ignore[no-any-return]

    @cached_property
    def dets(self) -> list[str]:
        """Returns a list of the detector names."""
        local_selection: list[str] = self.observation.select_local_detectors(
            selection=self.det_selection, flagmask=self.det_mask
        )
        return local_selection

    @property
    def focal_plane(self) -> toast.Focalplane:
        return self.observation.telescope.focalplane

    @property
    def sample_rate(self) -> float:
        """Returns the sampling rate (in Hz) of the data."""
        return self.focal_plane.sample_rate.to_value(u.Hz)  # type: ignore[no-any-return]

    def get_tods(self) -> Array:
        """Returns the timestream data."""
        # furax's LinearPolarizerOperator assumes power, TOAST assumes temperature
        return 0.5 * jnp.array(self.observation.detdata[self.det_data][self.dets, :])

    def get_pixels(self) -> Array:
        """Returns the pixel indices."""
        return jnp.array(self.observation.detdata[self.pixels][self.dets, :])

    def get_expanded_quats(self):  # type: ignore[no-untyped-def]
        """Returns expanded pointing quaternions."""
        return self.observation.detdata[self.quats][self.dets, :]

    def get_det_angles(self) -> Array:
        """Returns the detector angles on the sky."""
        func = np.vectorize(get_local_meridian_angle, signature='(n,k)->(n)')
        return jnp.array(func(self.get_expanded_quats()))

    def get_offsets(self) -> Array:
        """Returns the detector offset angles."""
        fp = self.focal_plane
        return jnp.array([fp[det]['gamma'].to_value(u.rad) for det in self.dets])

    def get_hwp_angles(self) -> Array:
        """Returns the HWP angles."""
        if self.hwp_angle is None:
            raise ValueError('HWP angle field not provided.')
        return jnp.array(self.observation.shared[self.hwp_angle].data)

    def get_psd_model(self) -> tuple[Array, Array]:
        """Returns frequencies and PSD values of the noise model."""
        if self.noise_model is None:
            raise ValueError('Noise model not provided.')
        model = self.observation[self.noise_model]
        freq = jnp.array([model.freq(det) for det in self.dets])
        psd = jnp.array([model.psd(det) for det in self.dets])
        return freq, psd