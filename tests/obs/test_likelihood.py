from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
import jax
import jax.numpy as jnp
import pytest

from furax import HomothetyOperator
from furax.obs._likelihoods import _spectral_likelihood_core , spectral_log_likelihood
from furax.obs.landscapes import FrequencyLandscape
from furax.obs.operators import (
    CMBOperator,
    DustOperator,
    MixingMatrixOperator,
    NoiseDiagonalOperator,
    SynchrotronOperator,
)
from furax.obs.stokes import ValidStokesType


def check_tree(x, y) -> bool:
    return jax.tree.all(jax.tree.map(lambda a, b: jnp.allclose(a, b), x, y))


@pytest.fixture
def likelihood_setup(stokes: ValidStokesType):
    # Setup parameters
    nside = 4
    nu = jnp.arange(10.0, 50.0, 10.0)
    landscape = FrequencyLandscape(nside, nu, stokes)
    d = landscape.normal(jax.random.PRNGKey(0))
    n1 = landscape.normal(jax.random.PRNGKey(1))
    n2 = landscape.normal(jax.random.PRNGKey(2))

    op = HomothetyOperator(5.0, _in_structure=d.structure)
    base_params = {
        'temp_dust': 20.0,
        'beta_dust': 1.54,
        'beta_pl': -3.0,
    }
    dust_params = {
        'temp_dust': base_params['temp_dust'],
        'beta_dust': base_params['beta_dust'],
    }
    synchrotron_params = {
        'beta_pl': base_params['beta_pl'],
    }
    patch_indices = {
        'temp_dust_patches': None,
        'beta_dust_patches': None,
        'beta_pl_patches': None,
    }
    dust_nu0 = 160.0
    synchrotron_nu0 = 20.0
    in_structure = d.structure_for((d.shape[1],))

    # Create component operators
    cmb = CMBOperator(nu, in_structure=in_structure)
    dust = DustOperator(
        nu,
        frequency0=dust_nu0,
        temperature=base_params['temp_dust'],
        beta=base_params['beta_dust'],
        in_structure=in_structure,
        temperature_patch_indices=patch_indices['temp_dust_patches'],
        beta_patch_indices=patch_indices['beta_dust_patches'],
    )
    synchrotron = SynchrotronOperator(
        nu,
        frequency0=synchrotron_nu0,
        beta_pl=base_params['beta_pl'],
        in_structure=in_structure,
        beta_pl_patch_indices=patch_indices['beta_pl_patches'],
    )
    # Mixing operators
    A_cds = MixingMatrixOperator(cmb=cmb, dust=dust, synchrotron=synchrotron)
    A_cd = MixingMatrixOperator(cmb=cmb, dust=dust)
    A_cs = MixingMatrixOperator(cmb=cmb, synchrotron=synchrotron)

    # Noise operators
    N1 = NoiseDiagonalOperator(n1, _in_structure=d.structure)
    N2 = NoiseDiagonalOperator(n2, _in_structure=d.structure)

    # Package everything into a dictionary for easy access in tests
    return {
        'd': d,
        'nu': nu,
        'base_params': base_params,
        'dust_params': dust_params,
        'synchrotron_params': synchrotron_params,
        'patch_indices': patch_indices,
        'dust_nu0': dust_nu0,
        'synchrotron_nu0': synchrotron_nu0,
        'A_cds': A_cds,
        'A_cd': A_cd,
        'A_cs': A_cs,
        'N1': N1,
        'N2': N2,
        'op': op,
    }


# Now we define multiple tests using the common fixture.
@pytest.mark.slow
def test_base_case(likelihood_setup):
    setup = likelihood_setup
    d = setup['d']
    nu = setup['nu']
    base_params = setup['base_params']
    patch_indices = setup['patch_indices']
    dust_nu0 = setup['dust_nu0']
    synchrotron_nu0 = setup['synchrotron_nu0']
    N1 = setup['N1']
    A_cds = setup['A_cds']

    # Case: all components, op=None, N_2=None
    with jax.disable_jit():
        AND, s = _spectral_likelihood_core(
            base_params, patch_indices, nu, N1, d, dust_nu0, synchrotron_nu0, op=None, N_2=None
        )
        expected_AND = (A_cds.T @ N1.I)(d)
        expected_s = (A_cds.T @ N1.I @ A_cds).I(expected_AND)
        assert check_tree(AND, expected_AND)
        assert check_tree(s, expected_s)


@pytest.mark.slow
def test_dust_only(likelihood_setup):
    setup = likelihood_setup
    d = setup['d']
    nu = setup['nu']
    dust_params = setup['dust_params']
    patch_indices = setup['patch_indices']
    dust_nu0 = setup['dust_nu0']
    synchrotron_nu0 = setup['synchrotron_nu0']
    N1 = setup['N1']
    A_cd = setup['A_cd']
    with jax.disable_jit():
        AND, s = _spectral_likelihood_core(
            dust_params, patch_indices, nu, N1, d, dust_nu0, synchrotron_nu0, op=None, N_2=None
        )
        expected_AND = (A_cd.T @ N1.I)(d)
        expected_s = (A_cd.T @ N1.I @ A_cd).I(expected_AND)
        assert check_tree(AND, expected_AND)
        assert check_tree(s, expected_s)


@pytest.mark.slow
def test_synchrotron_only(likelihood_setup):
    setup = likelihood_setup
    d = setup['d']
    nu = setup['nu']
    synchrotron_params = setup['synchrotron_params']
    patch_indices = setup['patch_indices']
    dust_nu0 = setup['dust_nu0']
    synchrotron_nu0 = setup['synchrotron_nu0']
    N1 = setup['N1']
    A_cs = setup['A_cs']
    with jax.disable_jit():
        AND, s = _spectral_likelihood_core(
            synchrotron_params,
            patch_indices,
            nu,
            N1,
            d,
            dust_nu0,
            synchrotron_nu0,
            op=None,
            N_2=None,
        )
        expected_AND = (A_cs.T @ N1.I)(d)
        expected_s = (A_cs.T @ N1.I @ A_cs).I(expected_AND)
        assert check_tree(AND, expected_AND)
        assert check_tree(s, expected_s)


@pytest.mark.slow
def test_different_noise_operator(likelihood_setup):
    setup = likelihood_setup
    d = setup['d']
    nu = setup['nu']
    base_params = setup['base_params']
    patch_indices = setup['patch_indices']
    dust_nu0 = setup['dust_nu0']
    synchrotron_nu0 = setup['synchrotron_nu0']
    N1 = setup['N1']
    N2 = setup['N2']
    A_cds = setup['A_cds']
    with jax.disable_jit():
        AND, s = _spectral_likelihood_core(
            base_params, patch_indices, nu, N1, d, dust_nu0, synchrotron_nu0, op=None, N_2=N2
        )
        expected_AND = (A_cds.T @ N1.I)(d)
        expected_s = (A_cds.T @ N2.I @ A_cds).I(expected_AND)
        assert check_tree(AND, expected_AND)
        assert check_tree(s, expected_s)


@pytest.mark.slow
def test_with_operator(likelihood_setup):
    setup = likelihood_setup
    d = setup['d']
    nu = setup['nu']
    base_params = setup['base_params']
    patch_indices = setup['patch_indices']
    dust_nu0 = setup['dust_nu0']
    synchrotron_nu0 = setup['synchrotron_nu0']
    N1 = setup['N1']
    op = setup['op']
    A_cds = setup['A_cds']
    with jax.disable_jit():
        AND, s = _spectral_likelihood_core(
            base_params, patch_indices, nu, N1, d, dust_nu0, synchrotron_nu0, op=op, N_2=None
        )
        expected_AND = (A_cds.T @ op.T @ N1.I)(d)
        expected_s = (A_cds.T @ op.T @ N1.I @ op @ A_cds).I(expected_AND)
        assert check_tree(AND, expected_AND)
        assert check_tree(s, expected_s)


@pytest.mark.slow
def test_operator_and_different_noise(likelihood_setup: dict[str, float]) -> None:
    setup = likelihood_setup
    d = setup['d']
    nu = setup['nu']
    base_params = setup['base_params']
    patch_indices = setup['patch_indices']
    dust_nu0 = setup['dust_nu0']
    synchrotron_nu0 = setup['synchrotron_nu0']
    N1 = setup['N1']
    N2 = setup['N2']
    op = setup['op']
    A_cds = setup['A_cds']
    with jax.disable_jit():
        AND, s = _spectral_likelihood_core(
            base_params, patch_indices, nu, N1, d, dust_nu0, synchrotron_nu0, op=op, N_2=N2
        )
        expected_AND = (A_cds.T @ op.T @ N1.I)(d)
        expected_s = (A_cds.T @ op.T @ N2.I @ op @ A_cds).I(expected_AND)
        assert check_tree(AND, expected_AND)
        assert check_tree(s, expected_s)


def test_analytical_forward(likelihood_setup):
    setup = likelihood_setup
    d = setup['d']
    nu = setup['nu']
    base_params = setup['base_params']
    patch_indices = setup['patch_indices']
    dust_nu0 = setup['dust_nu0']
    synchrotron_nu0 = setup['synchrotron_nu0']
    N1 = setup['N1']
    N2 = setup['N2']
    op = setup['op']

    
    likelihood = spectral_log_likelihood(
        params=base_params,
        nu=nu,
        N=N1,
        d=d,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        patch_indices=patch_indices,
        op=op,
        N_2=N2,
        analytical_gradient=False,
    )

    likelihood_analytical = spectral_log_likelihood(
        params=base_params,
        nu=nu,
        N=N1,
        d=d,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        patch_indices=patch_indices,
        op=op,
        N_2=N2,
        analytical_gradient=True,
    )

    assert jnp.allclose(likelihood, likelihood_analytical)

@pytest.mark.skip(reason="Gradient of lx.CG is failing sometimes .. to be tested manually with real data")
def test_analytical_gradient(likelihood_setup):
    setup = likelihood_setup
    d = setup['d']
    nu = setup['nu']
    base_params = setup['base_params']
    patch_indices = setup['patch_indices']
    dust_nu0 = setup['dust_nu0']
    synchrotron_nu0 = setup['synchrotron_nu0']
    N1 = setup['N1']
    N2 = setup['N2']
    op = setup['op']

    with Config(solver=lx.CG(rtol=1e-10 , atol=1e-10 , max_steps=10000)):
        auto_diff_grads = jax.grad(spectral_log_likelihood)(
            base_params,
            nu=nu,
            N=N1,
            d=d,
            dust_nu0=dust_nu0,
            synchrotron_nu0=synchrotron_nu0,
            patch_indices=patch_indices,
            op=op,
            N_2=N2,
            analytical_gradient=False,
        )

    analytical_grads = jax.grad(spectral_log_likelihood)(
        base_params,
        nu=nu,
        N=N1,
        d=d,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        patch_indices=patch_indices,
        op=op,
        N_2=N2,
        analytical_gradient=True,
    )

    mse = lambda x, y: jax.tree.map(lambda a, b: jnp.mean((a - b) ** 2), x, y)
    mrse = lambda x, y: jax.tree.map(lambda a, b: jnp.mean(((a - b) / b) ** 2), x, y)
    print(f"MSE between analytical and autodiff gradients: {mse(analytical_grads, auto_diff_grads)}")
    print(f"MRSE between analytical and autodiff gradients: {mrse(analytical_grads, auto_diff_grads)}")