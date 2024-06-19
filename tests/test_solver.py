import time

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from numpy.random import PCG64, Generator

from astrosim import Config
from astrosim.instruments.sat import (
    DETECTOR_ARRAY_SHAPE,
    NDIR_PER_DETECTOR,
    create_acquisition,
    create_detector_directions,
)
from astrosim.landscapes import HealpixLandscape, StokesIQUPyTree
from astrosim.operators import DiagonalOperator
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
    tod_structure = h.out_structure()
    coverage = h.T(jnp.ones(tod_structure.shape, tod_structure.dtype))
    m = DiagonalOperator(
        StokesIQUPyTree(
            I=coverage.I,
            Q=coverage.I,
            U=coverage.I,
        )
    ).I
    m = lx.TaggedLinearOperator(m, lx.positive_semidefinite_tag)
    hTh = lx.TaggedLinearOperator(h.T @ h, lx.positive_semidefinite_tag)
    tod = h(sky)
    b = h.T(tod)

    options = {'preconditioner': m}

    # solving
    time0 = time.time()
    solution = lx.linear_solve(
        hTh, b, solver=Config.instance().solver, throw=False, options=options
    )
    print(f'No JIT: {time.time() - time0}')
    assert solution.stats['num_steps'] < solution.stats['max_steps']

    @jax.jit
    def func(tod):
        return lx.linear_solve(
            hTh, b, solver=Config.instance().solver, throw=False, options=options
        )

    time0 = time.time()
    solution = func(tod)
    solution.value.I.block_until_ready()
    assert solution.stats['num_steps'] < solution.stats['max_steps']
    print(f'JIT 1:  {time.time() - time0}')

    del tod
    tod = h(sky)
    time0 = time.time()
    solution = func(tod)
    assert solution.stats['num_steps'] < solution.stats['max_steps']
    print(f'JIT 2:  {time.time() - time0}')

    with Config(solver_options=options):
        A = (h.T @ h).I @ h.T

    time0 = time.time()
    reconstructed_sky = A(tod)
    reconstructed_sky.I.block_until_ready()
    print('.I     ', time.time() - time0)

    @jax.jit
    def func2(tod):
        return A(tod)

    time0 = time.time()
    reconstructed_sky = func2(tod)
    reconstructed_sky.I.block_until_ready()
    print('JIT1 .I', time.time() - time0)

    time0 = time.time()
    reconstructed_sky = func2(tod)
    reconstructed_sky.I.block_until_ready()
    print('JIT2 .I', time.time() - time0)
