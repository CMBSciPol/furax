from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from furax.obs import CMBOperator, DustOperator, SynchrotronOperator
from furax.obs.landscapes import HealpixLandscape
from furax.obs.stokes import StokesPyTree


@pytest.fixture(scope='module')
def fg_data() -> tuple[dict[str, np.ndarray], StokesPyTree, jax.ShapeDtypeStruct]:
    fg_filename = Path(__file__).parent / 'data/fgbuster_data.npz'

    data = np.load(fg_filename)
    freq_maps = data['freq_maps']
    d = StokesPyTree.from_stokes(I=freq_maps[:, 0, :], Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :])

    nside = 32
    stokes_type = 'IQU'
    in_structure = HealpixLandscape(nside, stokes_type).structure

    return data, d, in_structure


def test_cmb_k_cmb(fg_data):
    data, d, in_structure = fg_data
    nu = data['frequencies']

    # Calculate CMB with K_CMB unit in furax
    cmb_fgbuster = data['CMB_K_CMB'][..., jnp.newaxis, jnp.newaxis] * data['freq_maps']
    cmb_fgbuster_tree = StokesPyTree.from_stokes(
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
    cmb_fgbuster_tree = StokesPyTree.from_stokes(
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
    dust_fgbuster_tree = StokesPyTree.from_stokes(
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
    dust_fgbuster_tree = StokesPyTree.from_stokes(
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
    synch_fgbuster_tree = StokesPyTree.from_stokes(
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
    synch_fgbuster_tree = StokesPyTree.from_stokes(
        I=synch_fgbuster[:, 0, :], Q=synch_fgbuster[:, 1, :], U=synch_fgbuster[:, 2, :]
    )

    synch_operator = SynchrotronOperator(
        nu, in_structure=in_structure, frequency0=20.0, units='K_RJ', beta_pl=-3.0
    )
    synch_furax = synch_operator(d)

    assert jax.tree.all(jax.tree.map(jnp.allclose, synch_furax, synch_fgbuster_tree))
