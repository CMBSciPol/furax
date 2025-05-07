import os
import subprocess
import sys
from typing import get_args

import healpy as hp
import jax
import numpy as np
import pytest
from _pytest import runner
from jaxtyping import Array, Float

from furax.obs.stokes import StokesIQU, ValidStokesType
from tests.helpers import TEST_DATA_PLANCK, TEST_DATA_SAT


@pytest.fixture(scope='session', autouse=True)
def enable_x64() -> None:
    jax.config.update('jax_enable_x64', True)


def load_planck(nside: int) -> np.array:
    PLANCK_URL = 'https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_143_2048_R3.01_full.fits'
    map_2048 = hp.read_map(PLANCK_URL, field=['I_STOKES', 'Q_STOKES', 'U_STOKES'])
    return hp.ud_grade(map_2048, nside)


@pytest.fixture(scope='session')
def planck_iqu_256() -> StokesIQU:
    nside = 256
    path = TEST_DATA_PLANCK / f'HFI_SkyMap_143_{nside}_R3.01_full_IQU.fits'
    if path.exists():
        maps = hp.read_map(path, field=[0, 1, 2])
    else:
        maps = load_planck(nside)
        TEST_DATA_PLANCK.mkdir(parents=True, exist_ok=True)
        hp.write_map(path, maps)
    i, q, u = maps.astype(float)
    return StokesIQU(
        i=jax.device_put(i),
        q=jax.device_put(q),
        u=jax.device_put(u),
    )


@pytest.fixture(scope='session')
def sat_nhits() -> Float[Array, '...']:
    nhits = hp.read_map(TEST_DATA_SAT / 'norm_nHits_SA_35FOV_G_nside512.fits').astype('<f8')
    npixel = nhits.size
    nhits[: npixel // 2] = 0
    nhits /= np.sum(nhits)
    return jax.device_put(nhits)


@pytest.fixture(params=get_args(ValidStokesType))
def stokes(request: pytest.FixtureRequest) -> ValidStokesType:
    """Parametrized fixture for I, QU, IQU and IQUV."""
    return request.param


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item):
    if not item.get_closest_marker('insubprocess'):
        return  # normal test: standard execution
    if os.environ.get('PYTEST_IN_SUBPROCESS') == '1':
        return
    ihook = item.ihook
    ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    reports = in_subprocess_run_report(item)
    for rep in reports:
        ihook.pytest_runtest_logreport(report=rep)
    ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def in_subprocess_run_report(item):
    EXITSTATUS_TESTEXIT = 4

    # Execution in subprocess
    env = os.environ.copy()
    env['PYTEST_IN_SUBPROCESS'] = '1'
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', item.nodeid],
            env=env,
            capture_output=True,
            encoding='utf-8',
        )
    except KeyboardInterrupt:
        os._exit(EXITSTATUS_TESTEXIT)

    keywords = {keyword: 1 for keyword in item.keywords}
    outcome = 'passed' if result.returncode == 0 else 'failed'
    sections = []
    if result.stdout:
        sections.append(('captured stdout', result.stdout))
    if result.stderr:
        sections.append(('captured stderr', result.stderr))
    reports = [
        runner.TestReport(
            item.nodeid,
            item.location,
            keywords,
            outcome,
            longrepr=None,
            when='call',
            sections=sections,
        )
    ]
    return reports
