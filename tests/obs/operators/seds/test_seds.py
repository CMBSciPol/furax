from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from furax.obs import CMBOperator, DustOperator, SynchrotronOperator
from furax.obs.landscapes import HealpixLandscape
from furax.obs.stokes import Stokes


@pytest.fixture(scope='module')
def fg_data() -> tuple[dict[str, np.ndarray], Stokes, jax.ShapeDtypeStruct]:
    fg_filename = Path(__file__).parent / 'data/fgbuster_data.npz'

    data = np.load(fg_filename)
    freq_maps = data['freq_maps']
    d = Stokes.from_stokes(I=freq_maps[:, 0, :], Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :])

    nside = 32
    stokes_type = 'IQU'
    in_structure = HealpixLandscape(nside, stokes_type).structure

    return data, d, in_structure


def test_cmb_k_cmb(fg_data):
    data, d, in_structure = fg_data
    nu = data['frequencies']

    # Calculate CMB with K_CMB unit in furax
    cmb_fgbuster = data['CMB_K_CMB'][..., jnp.newaxis, jnp.newaxis] * data['freq_maps']
    cmb_fgbuster_tree = Stokes.from_stokes(
        I=cmb_fgbuster[:, 0, :], Q=cmb_fgbuster[:, 1, :], U=cmb_fgbuster[:, 2, :]
    )

    cmb_operator = CMBOperator(nu, in_structure=in_structure, units='K_CMB')
    cmb_furax = cmb_operator(d)

    assert jax.tree.all(jax.tree.map(jnp.allclose, cmb_furax, cmb_fgbuster_tree))


def test_cmb_k_rj(fg_data):
    data, d, in_structure = fg_data
    nu = data['frequencies']

    # Calculate CMB with K_RJ unit in furax
    cmb_fgbuster = data['CMB_K_RJ'][..., jnp.newaxis, jnp.newaxis] * data['freq_maps']
    cmb_fgbuster_tree = Stokes.from_stokes(
        I=cmb_fgbuster[:, 0, :], Q=cmb_fgbuster[:, 1, :], U=cmb_fgbuster[:, 2, :]
    )

    cmb_operator = CMBOperator(nu, in_structure=in_structure, units='K_RJ')
    cmb_furax = cmb_operator(d)

    assert jax.tree.all(jax.tree.map(jnp.allclose, cmb_furax, cmb_fgbuster_tree))


def test_dust_k_cmb(fg_data):
    data, d, in_structure = fg_data
    nu = data['frequencies']

    # Calculate Dust with K_CMB unit in furax
    dust_fgbuster = data['DUST_K_CMB'][..., jnp.newaxis, jnp.newaxis] * data['freq_maps']
    dust_fgbuster_tree = Stokes.from_stokes(
        I=dust_fgbuster[:, 0, :], Q=dust_fgbuster[:, 1, :], U=dust_fgbuster[:, 2, :]
    )

    dust_operator = DustOperator(
        nu, in_structure=in_structure, frequency0=150.0, units='K_CMB', temperature=20.0, beta=1.54
    )
    dust_furax = dust_operator(d)

    assert jax.tree.all(jax.tree.map(jnp.allclose, dust_furax, dust_fgbuster_tree))


def test_dust_k_rj(fg_data):
    data, d, in_structure = fg_data
    nu = data['frequencies']

    # Calculate Dust with K_RJ unit in furax
    dust_fgbuster = data['DUST_K_RJ'][..., jnp.newaxis, jnp.newaxis] * data['freq_maps']
    dust_fgbuster_tree = Stokes.from_stokes(
        I=dust_fgbuster[:, 0, :], Q=dust_fgbuster[:, 1, :], U=dust_fgbuster[:, 2, :]
    )

    dust_operator = DustOperator(
        nu, in_structure=in_structure, frequency0=150.0, units='K_RJ', temperature=20.0, beta=1.54
    )
    dust_furax = dust_operator(d)

    assert jax.tree.all(jax.tree.map(jnp.allclose, dust_furax, dust_fgbuster_tree))


def test_synchrotron_k_cmb(fg_data):
    data, d, in_structure = fg_data
    nu = data['frequencies']

    # Calculate Synchrotron with K_CMB unit in furax
    synch_fgbuster = data['SYNC_K_CMB'][..., jnp.newaxis, jnp.newaxis] * data['freq_maps']
    synch_fgbuster_tree = Stokes.from_stokes(
        I=synch_fgbuster[:, 0, :], Q=synch_fgbuster[:, 1, :], U=synch_fgbuster[:, 2, :]
    )

    synch_operator = SynchrotronOperator(
        nu, in_structure=in_structure, frequency0=20.0, units='K_CMB', beta_pl=-3.0
    )
    synch_furax = synch_operator(d)

    assert jax.tree.all(jax.tree.map(jnp.allclose, synch_furax, synch_fgbuster_tree))


def test_synchrotron_k_rj(fg_data):
    data, d, in_structure = fg_data
    nu = data['frequencies']

    # Calculate Synchrotron with K_RJ unit in furax
    synch_fgbuster = data['SYNC_K_RJ'][..., jnp.newaxis, jnp.newaxis] * data['freq_maps']
    synch_fgbuster_tree = Stokes.from_stokes(
        I=synch_fgbuster[:, 0, :], Q=synch_fgbuster[:, 1, :], U=synch_fgbuster[:, 2, :]
    )

    synch_operator = SynchrotronOperator(
        nu, in_structure=in_structure, frequency0=20.0, units='K_RJ', beta_pl=-3.0
    )
    synch_furax = synch_operator(d)

    assert jax.tree.all(jax.tree.map(jnp.allclose, synch_furax, synch_fgbuster_tree))


def test_broadcasts_sky_map_without_frequency_axis(fg_data):
    """SED operators must broadcast a genuine (no-frequency) sky map to a multi-frequency cube.

    Regression guard: ``AbstractSEDOperator.__init__`` used to declare an ``in_structure`` with
    the frequency axis already baked in, making ``in_structure == out_structure`` and breaking
    this exact call (component-separation's real usage, e.g. via ``MixingMatrixOperator``).
    """
    data, _, in_structure = fg_data
    nu = data['frequencies']

    cmb_operator = CMBOperator(nu, in_structure=in_structure, units='K_CMB')

    # The declared in_structure must stay the plain sky map (no frequency axis).
    assert cmb_operator.in_structure == in_structure
    assert cmb_operator.out_structure.shape == (len(nu),) + in_structure.shape

    x = Stokes.from_stokes(
        I=jnp.arange(in_structure.shape[0], dtype=jnp.float64),
        Q=jnp.ones(in_structure.shape),
        U=-jnp.ones(in_structure.shape),
    )
    y = cmb_operator(x)

    assert y.shape == cmb_operator.out_structure.shape
    expected = cmb_operator.sed() * x.array[:, jnp.newaxis, :]
    assert jnp.allclose(y.array, expected)

    # Adjoint identity <Ax, z> == <x, A^T z>, guarding the custom mv() against a broken transpose.
    z = Stokes.from_stokes(
        *(jax.random.normal(jax.random.key(0), cmb_operator.out_structure.shape) for _ in range(3))
    )
    lhs = jnp.sum(y.array * z.array)
    rhs = jnp.sum(x.array * cmb_operator.T(z).array)
    assert jnp.allclose(lhs, rhs)
