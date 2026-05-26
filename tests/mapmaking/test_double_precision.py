"""Regression tests for ``MapMakingConfig.double_precision=False``.

Two layers of coverage:

1. **In-process dtype plumbing**: verifies that ``ObservationReader`` honours
   its ``dtype`` parameter and that ``MultiObservationMapMaker.get_reader``
   forwards ``config.dtype``. These run under the normal session-wide
   ``jax_enable_x64=True`` because they only inspect declared structures.

2. **Subprocess end-to-end run**: spawns a fresh Python process with
   ``JAX_ENABLE_X64=0`` (the configuration a real float32 user would use),
   reproduces a tiny mapmaker run under ``double_precision=False``, and
   asserts it completes with float32 outputs. The subprocess is required
   because ``tests/conftest.py`` installs a session-autouse fixture that
   forces ``jax_enable_x64=True`` for the whole pytest session; toggling
   that flag mid-session would also disturb every other test that depends
   on it.

See ``BUG_REPORT_double_precision_false.md`` for the underlying bug.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import jax.numpy as jnp
import pytest
from jaxtyping import Array, Bool, Float

from furax.mapmaking import (
    AbstractLazyObservation,
    AbstractSatelliteObservation,
    MultiObservationMapMaker,
    ObservationReader,
)
from furax.mapmaking.config import (
    HealpixConfig,
    LandscapeConfig,
    MapMakingConfig,
    Methods,
    NoiseConfig,
    PointingConfig,
)
from furax.mapmaking.noise import NoiseModel
from furax.obs.landscapes import ProjectionType, StokesLandscape

N_DETS = 1
N_SAMPS = 1024
SAMPLE_RATE = 100.0


class _FakeSatelliteObservation(AbstractSatelliteObservation[None]):
    """Self-contained synthetic observation used by the in-process tests.

    Returns arbitrary small arrays of whatever dtype the observation naturally
    produces (typically float64). The reader's ``dtype`` parameter is what
    decides the final dtype that flows into the operator chain.
    """

    def __init__(self) -> None:  # type: ignore[override]
        # bypass AbstractObservation.__init__ — we don't need an underlying
        # ``data`` container for these tests.
        pass

    @classmethod
    def from_file(cls, filename, requested_fields=None) -> _FakeSatelliteObservation:
        return cls()

    @property
    def name(self) -> str:
        return 'fake_obs'

    @property
    def telescope(self) -> str:
        return 'fake_telescope'

    @property
    def n_samples(self) -> int:
        return N_SAMPS

    @property
    def detectors(self) -> list[str]:
        return [f'det{i:02d}' for i in range(N_DETS)]

    @property
    def sample_rate(self) -> float:
        return SAMPLE_RATE

    def get_tods(self) -> Array:
        return jnp.zeros((N_DETS, N_SAMPS), dtype=jnp.float32)

    def get_detector_offset_angles(self) -> Array:
        return jnp.zeros(N_DETS, dtype=jnp.float64)

    def get_hwp_angles(self) -> Array:
        return jnp.zeros(N_SAMPS, dtype=jnp.float64)

    def get_sample_mask(self) -> Bool[Array, 'dets samps']:
        return jnp.ones((N_DETS, N_SAMPS), dtype=bool)

    def get_timestamps(self) -> Float[Array, ' a']:
        return jnp.asarray(1.7e9 + jnp.arange(N_SAMPS) / SAMPLE_RATE, dtype=jnp.float64)

    def get_wcs_shape_and_kernel(self, resolution_arcmin, projection=ProjectionType.CAR):
        raise NotImplementedError

    def get_pointing_and_spin_angles(self, landscape: StokesLandscape):
        raise NotImplementedError

    def get_noise_model(self) -> None | NoiseModel:
        return None

    def get_boresight_quaternions(self) -> Float[Array, 'samp 4']:
        q = jnp.zeros((N_SAMPS, 4), dtype=jnp.float64)
        return q.at[:, 0].set(1.0)

    def get_detector_quaternions(self) -> Float[Array, 'det 4']:
        q = jnp.zeros((N_DETS, 4), dtype=jnp.float64)
        return q.at[:, 0].set(1.0)


class _FakeLazyObservation(AbstractLazyObservation[None]):
    interface_class = _FakeSatelliteObservation

    def __init__(self) -> None:  # type: ignore[override]
        self.file = Path('<synthetic>')

    def get_data(self, requested_fields=None) -> _FakeSatelliteObservation:
        return _FakeSatelliteObservation()


# ----------------------------------------------------------------------------
# In-process dtype plumbing tests
# ----------------------------------------------------------------------------


REQUIRED_FIELDS = [
    'sample_data',
    'timestamps',
    'hwp_angles',
    'detector_quaternions',
    'boresight_quaternions',
]


class TestObservationReaderDtype:
    """Verify ObservationReader honours its ``dtype`` parameter."""

    def test_default_dtype_is_float64(self) -> None:
        """Backwards compatibility: no dtype argument → float64 structures.

        External callers that construct an ObservationReader directly (i.e. not
        via MultiObservationMapMaker) must continue to see float64, otherwise
        we would silently break the historical contract.
        """
        reader = ObservationReader([_FakeLazyObservation()], requested_fields=REQUIRED_FIELDS)
        assert reader.dtype == jnp.float64
        for field in REQUIRED_FIELDS:
            assert reader.out_structure[field].dtype == jnp.float64, (
                f'expected float64 for {field}, got {reader.out_structure[field].dtype}'
            )

    def test_explicit_float32_propagates_to_all_float_fields(self) -> None:
        """``dtype=jnp.float32`` must apply to every float ShapeDtypeStruct.

        The original fix attempt only touched ``sample_data``; under
        ``jax_enable_x64=False`` JAX rejects any float64 ShapeDtypeStruct,
        so timestamps/HWP/quaternions/noise_model_fits must also be float32.
        """
        reader = ObservationReader(
            [_FakeLazyObservation()],
            requested_fields=REQUIRED_FIELDS,
            dtype=jnp.float32,
        )
        assert reader.dtype == jnp.float32
        for field in REQUIRED_FIELDS:
            assert reader.out_structure[field].dtype == jnp.float32, (
                f'expected float32 for {field}, got {reader.out_structure[field].dtype}'
            )

    def test_bool_masks_remain_bool(self) -> None:
        """Sanity check: changing ``dtype`` must not affect boolean masks."""
        reader = ObservationReader(
            [_FakeLazyObservation()],
            requested_fields=['valid_sample_masks', 'sample_data'],
            dtype=jnp.float32,
        )
        assert reader.out_structure['valid_sample_masks'].dtype == jnp.bool


class TestMapMakerForwardsDtype:
    """``MultiObservationMapMaker.get_reader`` must use ``config.dtype``."""

    @pytest.mark.parametrize(
        ('double_precision', 'expected_dtype'),
        [(True, jnp.float64), (False, jnp.float32)],
    )
    def test_get_reader_uses_config_dtype(self, double_precision: bool, expected_dtype) -> None:
        config = MapMakingConfig(
            method=Methods.BINNED,
            landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=8)),
            noise=NoiseConfig(white=True, fit_from_data=True),
            pointing=PointingConfig(on_the_fly=True),
            double_precision=double_precision,
        )
        maker = MultiObservationMapMaker([_FakeLazyObservation()], config=config)
        reader = maker.get_reader(REQUIRED_FIELDS)
        assert reader.dtype == expected_dtype
        for field in REQUIRED_FIELDS:
            assert reader.out_structure[field].dtype == expected_dtype


# ----------------------------------------------------------------------------
# Subprocess end-to-end test
# ----------------------------------------------------------------------------


_SUBPROCESS_SCRIPT = textwrap.dedent("""\
    import os, sys
    # Sanity check: JAX_ENABLE_X64 must be set BEFORE importing jax.
    assert os.environ.get('JAX_ENABLE_X64') == '0'

    import jax
    # jax_healpy.__init__ unconditionally flips x64 back to True on import.
    # We therefore must reset it AFTER all furax-side imports finish.
    jax.config.update('jax_enable_x64', False)

    import jax.numpy as jnp
    from pathlib import Path
    import numpy as np
    from jaxtyping import Array, Bool, Float

    from furax.mapmaking import (
        AbstractLazyObservation, AbstractSatelliteObservation,
        MultiObservationMapMaker,
    )
    from furax.mapmaking.config import (
        HealpixConfig, LandscapeConfig, MapMakingConfig, Methods,
        NoiseConfig, NoiseFitConfig, PointingConfig, SolverConfig,
    )
    from furax.obs.landscapes import ProjectionType, StokesLandscape
    from furax.mapmaking.noise import NoiseModel

    # Flip back to False (jax_healpy turned it on during furax imports).
    jax.config.update('jax_enable_x64', False)
    assert not jax.config.read('jax_enable_x64'), 'x64 must be OFF for this test'

    N_DETS, N_SAMPS, SAMPLE_RATE = 1, 1024, 100.0

    class _Obs(AbstractSatelliteObservation):
        def __init__(self): pass
        @classmethod
        def from_file(cls, *a, **kw): return cls()
        @property
        def name(self): return 'fake'
        @property
        def telescope(self): return 'fake'
        @property
        def n_samples(self): return N_SAMPS
        @property
        def detectors(self): return ['d0']
        @property
        def sample_rate(self): return SAMPLE_RATE
        def get_tods(self):
            return jnp.asarray(np.random.default_rng(0).normal(size=(N_DETS, N_SAMPS)).astype(np.float32))
        def get_detector_offset_angles(self): return jnp.zeros(N_DETS)
        def get_hwp_angles(self):
            t = np.arange(N_SAMPS) / SAMPLE_RATE
            return jnp.asarray((2*np.pi*2.0*t) % (2*np.pi))
        def get_sample_mask(self): return jnp.ones((N_DETS, N_SAMPS), dtype=bool)
        def get_timestamps(self):
            return jnp.asarray(1.7e9 + np.arange(N_SAMPS) / SAMPLE_RATE)
        def get_wcs_shape_and_kernel(self, *a, **kw): raise NotImplementedError
        def get_pointing_and_spin_angles(self, *a, **kw): raise NotImplementedError
        def get_noise_model(self): return None
        def get_boresight_quaternions(self):
            phi = np.linspace(0.0, np.pi/4, N_SAMPS)
            q = np.zeros((N_SAMPS, 4))
            q[:, 0] = np.cos(phi/2); q[:, 3] = np.sin(phi/2)
            return jnp.asarray(q)
        def get_detector_quaternions(self):
            q = np.zeros((N_DETS, 4)); q[:, 0] = 1.0
            return jnp.asarray(q)

    class _Lazy(AbstractLazyObservation):
        interface_class = _Obs
        def __init__(self): self.file = Path('<x>')
        def get_data(self, *a, **kw): return _Obs()

    config = MapMakingConfig(
        method=Methods.BINNED,
        landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=8)),
        noise=NoiseConfig(
            white=True, fit_from_data=True,
            fitting=NoiseFitConfig(nperseg=256, mask_hwp_harmonics=False),
        ),
        pointing=PointingConfig(on_the_fly=True, chunk_size=32, interpolation='nearest'),
        double_precision=False,
        scanning_mask=False, sample_mask=False, hits_cut=0.0, cond_cut=0.0,
        solver=SolverConfig(rtol=1e-6, atol=0, max_steps=10),
    )
    maker = MultiObservationMapMaker([_Lazy()], config=config)
    results = maker.run()

    # Acceptance checks: ran without raising AND outputs are float32.
    map_dtype = jax.tree.leaves(results.map)[0].dtype
    icov_dtype = results.icov.dtype
    print(f'OK map={map_dtype} icov={icov_dtype}')
    assert map_dtype == jnp.float32, f'expected float32 map, got {map_dtype}'
    assert icov_dtype == jnp.float32, f'expected float32 icov, got {icov_dtype}'
""")


@pytest.mark.slow
def test_mapmaker_runs_under_double_precision_false_and_x64_off() -> None:
    """End-to-end regression test for the bug.

    Runs in a subprocess so it can have ``jax_enable_x64=False`` without
    fighting the session-autouse ``enable_x64`` fixture in
    ``tests/conftest.py``.
    """
    env = {**__import__('os').environ, 'JAX_ENABLE_X64': '0'}
    result = subprocess.run(
        [sys.executable, '-c', _SUBPROCESS_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert result.returncode == 0, (
        f'subprocess failed with code {result.returncode}.\n'
        f'--- stdout ---\n{result.stdout}\n'
        f'--- stderr ---\n{result.stderr}'
    )
    assert 'OK map=float32 icov=float32' in result.stdout, (
        f'subprocess did not print the expected acceptance line.\n'
        f'--- stdout ---\n{result.stdout}\n'
        f'--- stderr ---\n{result.stderr}'
    )
