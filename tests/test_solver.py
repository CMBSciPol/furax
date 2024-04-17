import time

import jax
import lineax as lx
import numpy as np
import pytest
from numpy.random import PCG64, Generator

from astrosim.instruments.sat import (
    DETECTOR_ARRAY_SHAPE,
    NDIR_PER_DETECTOR,
    create_acquisition,
    create_detector_directions,
)
from astrosim.landscapes import HealpixLandscape, StokesIQUPyTree
from astrosim.operators.projections import PytreeDiagonalOperator
from astrosim.samplings import create_random_sampling


def get_random_generator(seed: int) -> np.random.Generator:
    return Generator(PCG64(seed))


NSIDE = 256
RANDOM_SEED = 0
RANDOM_GENERATOR = get_random_generator(RANDOM_SEED)
NSAMPLING = 1_000


@pytest.mark.slow
def test_solver(planck_iqu_256, sat_nhits):
    print(f'#SAMPLINGS: {NSAMPLING}')
    print(f'#DETECTORS: {np.prod(DETECTOR_ARRAY_SHAPE)}')
    print(f'#NDIRS: {NDIR_PER_DETECTOR}')

    sky = planck_iqu_256

    # sky model
    landscape = HealpixLandscape(NSIDE, 'IQU')

    # instrument model
    random_generator = get_random_generator(RANDOM_SEED)
    samplings = create_random_sampling(sat_nhits, NSAMPLING, random_generator)
    detector_dirs = create_detector_directions()
    h = create_acquisition(landscape, samplings, detector_dirs)

    # preconditioner
    coverage_highres = landscape.get_coverage(samplings)
    m_diagonal = landscape.ones()
    mask = coverage_highres > 0
    m_diagonal = StokesIQUPyTree(
        I=m_diagonal.I.at[mask].set(1 / coverage_highres[mask]),
        Q=m_diagonal.Q,
        U=m_diagonal.U,
    )
    m = PytreeDiagonalOperator(m_diagonal)

    # solving
    hTh = lx.TaggedLinearOperator(h.T @ h, lx.positive_semidefinite_tag)
    m = lx.TaggedLinearOperator(m, lx.positive_semidefinite_tag)
    tod = h(sky)
    b = h.T(tod)
    solver = lx.CG(rtol=1e-4, atol=1e-4, max_steps=1000)

    time0 = time.time()
    solution = lx.linear_solve(hTh, b, solver=solver, throw=False, options={'preconditioner': m})
    print(f'No JIT: {time.time() - time0}')
    assert solution.stats['num_steps'] < solution.stats['max_steps']

    @jax.jit
    def func(tod):
        return lx.linear_solve(hTh, b, solver=solver, throw=False, options={'preconditioner': m})

    time0 = time.time()
    solution = func(tod)
    assert solution.stats['num_steps'] < solution.stats['max_steps']
    print(f'JIT 1: {time.time() - time0}')

    del tod
    tod = h(sky)
    time0 = time.time()
    solution = func(tod)
    assert solution.stats['num_steps'] < solution.stats['max_steps']
    print(f'JIT 2: {time.time() - time0}')
