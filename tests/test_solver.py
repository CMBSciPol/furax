import time

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from numpy.random import PCG64, Generator

from furax import Config, DiagonalOperator
from furax._instruments.sat import (
    DETECTOR_ARRAY_SHAPE,
    NDIR_PER_DETECTOR,
    create_acquisition,
    create_detector_directions,
)
from furax.obs._samplings import create_random_sampling
from furax.obs.landscapes import HealpixLandscape
from furax.tree import as_structure


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
    h = h.reduce()

    # preconditioner
    tod_structure = h.out_structure()
    coverage = h.T(jnp.ones(tod_structure.shape, tod_structure.dtype))
    m = DiagonalOperator(coverage.i, in_structure=as_structure(coverage)).I
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
    jax.block_until_ready(solution.value)
    print(f'No JIT: {time.time() - time0}')
    assert solution.stats['num_steps'] < solution.stats['max_steps']

    # import healpy as hp
    # import matplotlib.pyplot as mp
    #
    # hp.mollview(solution.value.i)
    # mp.show()

    @jax.jit
    def func(tod):
        b = h.T(tod)
        return lx.linear_solve(
            hTh, b, solver=Config.instance().solver, throw=False, options=options
        )

    time0 = time.time()
    solution = func(tod)
    jax.block_until_ready(solution.value)
    assert solution.stats['num_steps'] < solution.stats['max_steps']
    print(f'JIT 1:  {time.time() - time0}')

    del tod
    tod = h(sky)
    time0 = time.time()
    solution = func(tod)
    jax.block_until_ready(solution.value)
    assert solution.stats['num_steps'] < solution.stats['max_steps']
    print(f'JIT 2:  {time.time() - time0}')

    with Config(solver_options=options):
        A = (h.T @ h).I @ h.T

    time0 = time.time()
    reconstructed_sky = A(tod)
    jax.block_until_ready(reconstructed_sky)
    print('.I     ', time.time() - time0)

    @jax.jit
    def func2(tod):
        return A(tod)

    time0 = time.time()
    reconstructed_sky = func2(tod)
    jax.block_until_ready(reconstructed_sky)
    print('JIT 1 .I', time.time() - time0)

    time0 = time.time()
    reconstructed_sky = func2(tod)
    jax.block_until_ready(reconstructed_sky)
    print('JIT 2 .I', time.time() - time0)
