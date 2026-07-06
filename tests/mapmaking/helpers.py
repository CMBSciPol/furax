from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Float

from furax.mapmaking import (
    AbstractGroundObservation,
    AbstractLazyObservation,
    AbstractObservation,
)
from furax.mapmaking.noise import AtmosphericNoiseModel, NoiseModel
from furax.obs.landscapes import ProjectionType, StokesLandscape
from furax.obs.stokes import Stokes, ValidStokesType


class FakeObservation(AbstractObservation[None]):
    """Self-contained, file-free synthetic telescope observation.

    Like the real interfaces, the getters return host (numpy) arrays; the
    reader moves them to device through ``io_callback`` and the reader's
    ``dtype`` parameter decides the final dtype flowing into the operator
    chain. Geometry (boresight sweep, HWP) is parameterised so the same
    factory can stand in for a range of observations.

    Supports both the modulated and the demodulated reader paths: the
    ``get_demodulated_*`` getters return synthetic per-Stokes streams and
    white-noise fits, enough to exercise the demodulated pipeline (which
    otherwise needs sotodlib + ``.h5`` fixtures) but not for numerical
    validation.

    Only the on-the-fly pointing path is supported:
    ``get_wcs_shape_and_kernel`` and ``get_pointing_and_spin_angles`` raise
    ``NotImplementedError`` (use ``PointingConfig(on_the_fly=True)`` and a
    healpix landscape).
    """

    def __init__(
        self,
        *,
        n_dets: int = 1,
        n_samples: int = 1024,
        sample_rate: float = 100.0,
        hwp_frequency: float = 2.0,
        seed: int = 0,
    ) -> None:  # type: ignore[override]
        # Bypass AbstractObservation.__init__: there is no underlying
        # ``data`` container for an in-memory observation.
        self._n_dets = n_dets
        self._n_samples = n_samples
        self._sample_rate = sample_rate
        self._hwp_frequency = hwp_frequency
        self._seed = seed

    @classmethod
    def from_file(cls, filename, requested_fields=None) -> FakeObservation:
        return cls()

    @property
    def name(self) -> str:
        return 'fake_obs'

    @property
    def telescope(self) -> str:
        return 'fake_telescope'

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def detectors(self) -> list[str]:
        return [f'det{i:02d}' for i in range(self._n_dets)]

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    def get_tods(self) -> Float[np.ndarray, 'dets samps']:
        # Non-zero data so the white-noise PSD fit yields a finite sigma.
        rng = np.random.default_rng(self._seed)
        return rng.normal(size=(self._n_dets, self._n_samples)).astype(np.float32)

    def get_demodulated_tods(self, stokes: ValidStokesType = 'IQU') -> Any:
        # One synthetic (dets, samps) stream per requested Stokes component.
        kls = Stokes.class_for(stokes)
        rng = np.random.default_rng(self._seed + 1)
        streams = [
            rng.normal(size=(self._n_dets, self._n_samples)).astype(np.float32) for _ in stokes
        ]
        return kls.from_stokes(*streams)

    def get_demodulated_noise_model(self, stokes: ValidStokesType = 'IQU') -> Any:
        # One white-noise fit per Stokes component. Columns match
        # AtmosphericNoiseModel.to_array: (sigma, alpha, fk, f0).
        kls = Stokes.class_for(stokes)
        n = self._n_dets
        fit = np.column_stack([np.ones(n), np.ones(n), np.ones(n), 1e-5 * np.ones(n)]).astype(
            np.float64
        )
        return kls.from_stokes(*[fit for _ in stokes])

    def get_detector_offset_angles(self) -> Float[np.ndarray, ' dets']:
        return np.zeros(self._n_dets, dtype=np.float64)

    def get_hwp_angles(self) -> Float[np.ndarray, ' a']:
        # Sweeping HWP angle wrapped to [0, 2pi).
        t = np.arange(self._n_samples) / self._sample_rate
        return np.asarray((2 * np.pi * self._hwp_frequency * t) % (2 * np.pi), dtype=np.float64)

    def get_sample_mask(self) -> Bool[np.ndarray, 'dets samps']:
        return np.ones((self._n_dets, self._n_samples), dtype=bool)

    def get_timestamps(self) -> Float[np.ndarray, ' a']:
        return np.asarray(1.7e9 + np.arange(self._n_samples) / self._sample_rate, dtype=np.float64)

    def get_wcs_shape_and_kernel(self, resolution_arcmin, projection=ProjectionType.CAR):
        raise NotImplementedError

    def get_pointing_and_spin_angles(self, landscape: StokesLandscape):
        raise NotImplementedError

    def get_noise_model(self) -> None | NoiseModel:
        return None

    def get_boresight_quaternions(self) -> Float[np.ndarray, 'samp 4']:
        # Sweep the boresight so samples hit a range of sky pixels.
        phi = np.linspace(0.0, np.pi / 4, self._n_samples)
        q = np.zeros((self._n_samples, 4), dtype=np.float64)
        q[:, 0] = np.cos(phi / 2)
        q[:, 3] = np.sin(phi / 2)
        return q

    def get_detector_quaternions(self) -> Float[np.ndarray, 'det 4']:
        q = np.zeros((self._n_dets, 4), dtype=np.float64)
        q[:, 0] = 1.0
        return q


class FakeLazyObservation(AbstractLazyObservation[None]):
    """File-free lazy synthetic observation.

    Accepted anywhere a real ``AbstractLazyObservation`` is (e.g.
    ``MultiObservationMapMaker([...])`` or
    ``ObservationReader.from_observations([...])``). Only the on-the-fly pointing
    path with a healpix landscape is supported. ``kwargs`` are forwarded to
    :class:`FakeObservation` (``n_dets``, ``n_samples``,
    ``sample_rate``, ``hwp_frequency``, ``seed``).
    """

    interface_class = FakeObservation

    def __init__(self, **kwargs: Any) -> None:  # type: ignore[override]
        self.file = Path('<synthetic>')
        self._kwargs = kwargs

    def get_data(self, requested_fields=None) -> FakeObservation:
        return FakeObservation(**self._kwargs)


class FailingLazyObservation(FakeLazyObservation):
    """Lazy obs whose ``get_data`` always raises (a deterministic preprocessing failure).

    The shape probe still succeeds (failures are mainly in the full load), so the mapmaker sizes
    its buffers normally and must gate this observation out at read time.
    """

    @property
    def name(self) -> str:
        return 'failing_obs'

    def probe_shape(self) -> tuple[int, int]:
        obs = FakeObservation(**self._kwargs)
        return len(obs.detectors), obs.n_samples

    def get_data(self, requested_fields=None) -> FakeObservation:
        raise RuntimeError('simulated preprocessing failure')


class GappyGroundObservation(FakeObservation, AbstractGroundObservation[None]):
    """Ground observation with flagged gaps in the sample mask and a finite 1/f noise model.

    Use ``weighting.source = PRECOMPUTED`` to avoid fitting a 1/f model to synthetic white TODs.
    """

    def get_scanning_intervals(self) -> Float[np.ndarray, 'n 2']:
        half = self._n_samples // 2
        return np.array([[0, half], [half, self._n_samples]])

    def get_elapsed_times(self) -> Float[np.ndarray, ' a']:
        return np.arange(self._n_samples, dtype=np.float64) / self._sample_rate

    def get_azimuth(self) -> Float[np.ndarray, ' a']:
        return np.linspace(0.0, 10.0, self._n_samples).astype(np.float64)

    def get_elevation(self) -> Float[np.ndarray, ' a']:
        return np.full(self._n_samples, 0.8, dtype=np.float64)

    def get_left_scan_mask(self) -> Float[np.ndarray, ' a']:
        return (np.arange(self._n_samples) % 2 == 0).astype(np.float64)

    def get_right_scan_mask(self) -> Float[np.ndarray, ' a']:
        return (np.arange(self._n_samples) % 2 == 1).astype(np.float64)

    def get_noise_model(self) -> NoiseModel:
        n = self._n_dets
        return AtmosphericNoiseModel(
            sigma=jnp.ones(n),
            alpha=-jnp.ones(n),
            fk=0.1 * jnp.ones(n),
            f0=1e-3 * jnp.ones(n),
        )

    def get_sample_mask(self) -> Bool[np.ndarray, 'dets samps']:
        mask = np.ones((self._n_dets, self._n_samples), dtype=bool)
        gap = max(2, self._n_samples // 20)
        for centre in (self._n_samples // 4, self._n_samples // 2, 3 * self._n_samples // 4):
            mask[:, centre - gap // 2 : centre + gap // 2] = False
        return mask


class GappyLazyGroundObservation(FakeLazyObservation):
    """File-free lazy wrapper around :class:`GappyGroundObservation`."""

    interface_class = GappyGroundObservation

    def get_data(self, requested_fields=None) -> GappyGroundObservation:
        return GappyGroundObservation(**self._kwargs)

    def probe_shape(self) -> tuple[int, int]:
        obs = GappyGroundObservation(**self._kwargs)
        return len(obs.detectors), obs.n_samples


class FakeGroundObservation(FakeObservation, AbstractGroundObservation[None]):
    """``FakeObservation`` extended with the ground getters the single-observation
    ``MapMaker`` template path needs (azimuth/elevation, scanning intervals, scan masks),
    plus a finite precomputed noise model.

    The precomputed ``AtmosphericNoiseModel`` sidesteps fitting a 1/f model to the synthetic
    white TODs (which yields NaN); use ``weighting.source = PRECOMPUTED``.
    """

    def get_scanning_intervals(self) -> Float[np.ndarray, 'n 2']:
        half = self._n_samples // 2
        return np.array([[0, half], [half, self._n_samples]])

    def get_elapsed_times(self) -> Float[np.ndarray, ' a']:
        return np.arange(self._n_samples, dtype=np.float64) / self._sample_rate

    def get_azimuth(self) -> Float[np.ndarray, ' a']:
        return np.linspace(0.0, 10.0, self._n_samples).astype(np.float64)

    def get_elevation(self) -> Float[np.ndarray, ' a']:
        return np.full(self._n_samples, 0.8, dtype=np.float64)

    def get_left_scan_mask(self) -> Float[np.ndarray, ' a']:
        return (np.arange(self._n_samples) % 2 == 0).astype(np.float64)

    def get_right_scan_mask(self) -> Float[np.ndarray, ' a']:
        return (np.arange(self._n_samples) % 2 == 1).astype(np.float64)

    def get_noise_model(self) -> NoiseModel:
        n = self._n_dets
        return AtmosphericNoiseModel(
            sigma=jnp.ones(n),
            alpha=-jnp.ones(n),
            fk=0.1 * jnp.ones(n),
            f0=1e-3 * jnp.ones(n),
        )
