from __future__ import annotations

from functools import lru_cache
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from furax.mapmaking import AbstractLazyObservation, AbstractSatelliteObservation

TEST_DATA = Path(__file__).parent / 'data'
TEST_DATA_PLANCK = TEST_DATA / 'planck'
TEST_DATA_SAT = TEST_DATA / 'sat'


def arange(*shape: int, dtype=jnp.float32, start=1) -> jax.Array:
    """arange(2, 3) -> jnp.arange(6, dtype=jnp.float32).reshape(2, 3)"""
    return jnp.arange(start, prod(shape) + start, dtype=dtype).reshape(shape)


@lru_cache(maxsize=1)
def _fake_observation_classes() -> tuple[type, type]:
    """Build the synthetic observation classes on first use.

    The classes are defined inside a cached function (rather than at module top
    level) so that importing ``tests.helpers`` -- which the root ``conftest.py``
    does for *every* test session -- does not pull in ``furax.mapmaking`` (a
    ~1.6s import) for unrelated tests. ``lru_cache`` guarantees the classes are
    defined exactly once, keeping ``isinstance`` / ``interface_class`` identity
    stable across calls.
    """
    import numpy as np
    from jaxtyping import Bool, Float

    from furax.mapmaking import AbstractLazyObservation, AbstractSatelliteObservation
    from furax.mapmaking.noise import NoiseModel
    from furax.obs.landscapes import ProjectionType, StokesLandscape
    from furax.obs.stokes import Stokes

    class FakeSatelliteObservation(AbstractSatelliteObservation[None]):
        """Self-contained, file-free synthetic satellite observation.

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
        def from_file(cls, filename, requested_fields=None) -> FakeSatelliteObservation:
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

        def get_demodulated_tods(self, stokes: str = 'IQU') -> Any:
            # One synthetic (dets, samps) stream per requested Stokes component.
            kls = Stokes.class_for(stokes)
            rng = np.random.default_rng(self._seed + 1)
            streams = [
                rng.normal(size=(self._n_dets, self._n_samples)).astype(np.float32) for _ in stokes
            ]
            return kls.from_stokes(*streams)

        def get_demodulated_noise_models(self, stokes: str = 'IQU') -> Any:
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
            return np.asarray(
                1.7e9 + np.arange(self._n_samples) / self._sample_rate, dtype=np.float64
            )

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
        interface_class = FakeSatelliteObservation

        def __init__(self, **kwargs: Any) -> None:  # type: ignore[override]
            self.file = Path('<synthetic>')
            self._kwargs = kwargs

        def get_data(self, requested_fields=None) -> FakeSatelliteObservation:
            return FakeSatelliteObservation(**self._kwargs)

    return FakeSatelliteObservation, FakeLazyObservation


def make_fake_satellite_observation(
    *,
    n_dets: int = 1,
    n_samples: int = 1024,
    sample_rate: float = 100.0,
    hwp_frequency: float = 2.0,
    seed: int = 0,
) -> AbstractSatelliteObservation[None]:
    """Return a self-contained, file-free synthetic satellite observation.

    See :func:`make_fake_lazy_observation` for the lazy wrapper accepted by
    ``MultiObservationMapMaker`` / ``ObservationReader.from_observations``.
    """
    cls, _ = _fake_observation_classes()
    return cls(
        n_dets=n_dets,
        n_samples=n_samples,
        sample_rate=sample_rate,
        hwp_frequency=hwp_frequency,
        seed=seed,
    )


def make_fake_lazy_observation(
    *,
    n_dets: int = 1,
    n_samples: int = 1024,
    sample_rate: float = 100.0,
    hwp_frequency: float = 2.0,
    seed: int = 0,
) -> AbstractLazyObservation[None]:
    """Return a file-free lazy synthetic observation.

    Accepted anywhere a real ``AbstractLazyObservation`` is (e.g.
    ``MultiObservationMapMaker([...])`` or
    ``ObservationReader.from_observations([...])``). Only the on-the-fly pointing
    path with a healpix landscape is supported.
    """
    _, cls = _fake_observation_classes()
    return cls(
        n_dets=n_dets,
        n_samples=n_samples,
        sample_rate=sample_rate,
        hwp_frequency=hwp_frequency,
        seed=seed,
    )
