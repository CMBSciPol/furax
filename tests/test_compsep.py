import pytest
import jax
import jax.numpy as jnp
from furax.operators.seds import CMBOperator, DustOperator, SynchrotronOperator
from furax.landscapes import StokesPyTree
from jaxtyping import Array


def test_cmb_k_cmb(get_fgbuster_data):
    fg_data, d, in_structure = get_fgbuster_data
    nu = fg_data['frequencies']
    
    # Calculate CMB with K_CMB unit in furax
    cmb_fgbuster = fg_data['CMB_K_CMB'][..., jnp.newaxis, jnp.newaxis] * fg_data['freq_maps']
    cmb_fgbuster_tree = StokesPyTree.from_stokes(I=cmb_fgbuster[:, 0, :],
                                                 Q=cmb_fgbuster[:, 1, :],
                                                 U=cmb_fgbuster[:, 2, :])
    
    cmb_operator = CMBOperator(nu, in_structure=in_structure, units='K_CMB')
    cmb_furax = cmb_operator(d)
    
    assert jax.tree.all(jax.tree.map(jnp.allclose, cmb_furax, cmb_fgbuster_tree))


def test_cmb_k_rj(get_fgbuster_data):
    fg_data, d, in_structure = get_fgbuster_data
    nu = fg_data['frequencies']
    
    # Calculate CMB with K_RJ unit in furax
    cmb_fgbuster = fg_data['CMB_K_RJ'][..., jnp.newaxis, jnp.newaxis] * fg_data['freq_maps']
    cmb_fgbuster_tree = StokesPyTree.from_stokes(I=cmb_fgbuster[:, 0, :],
                                                 Q=cmb_fgbuster[:, 1, :],
                                                 U=cmb_fgbuster[:, 2, :])
    
    cmb_operator = CMBOperator(nu, in_structure=in_structure, units='K_RJ')
    cmb_furax = cmb_operator(d)
    
    assert jax.tree.all(jax.tree.map(jnp.allclose, cmb_furax, cmb_fgbuster_tree))


def test_dust_k_cmb(get_fgbuster_data):
    fg_data, d, in_structure = get_fgbuster_data
    nu = fg_data['frequencies']
    
    # Calculate Dust with K_CMB unit in furax
    dust_fgbuster = fg_data['DUST_K_CMB'][..., jnp.newaxis, jnp.newaxis] * fg_data['freq_maps']
    dust_fgbuster_tree = StokesPyTree.from_stokes(I=dust_fgbuster[:, 0, :],
                                                  Q=dust_fgbuster[:, 1, :],
                                                  U=dust_fgbuster[:, 2, :])
    
    dust_operator = DustOperator(nu, in_structure=in_structure, frequency0=150., units='K_CMB', temperature=20., beta=1.54)
    dust_furax = dust_operator(d)
    
    assert jax.tree.all(jax.tree.map(jnp.allclose, dust_furax, dust_fgbuster_tree))


def test_dust_k_rj(get_fgbuster_data):
    fg_data, d, in_structure = get_fgbuster_data
    nu = fg_data['frequencies']
    
    # Calculate Dust with K_RJ unit in furax
    dust_fgbuster = fg_data['DUST_K_RJ'][..., jnp.newaxis, jnp.newaxis] * fg_data['freq_maps']
    dust_fgbuster_tree = StokesPyTree.from_stokes(I=dust_fgbuster[:, 0, :],
                                                  Q=dust_fgbuster[:, 1, :],
                                                  U=dust_fgbuster[:, 2, :])
    
    dust_operator = DustOperator(nu, in_structure=in_structure, frequency0=150., units='K_RJ', temperature=20., beta=1.54)
    dust_furax = dust_operator(d)
    
    assert jax.tree.all(jax.tree.map(jnp.allclose, dust_furax, dust_fgbuster_tree))


def test_synchrotron_k_cmb(get_fgbuster_data):
    fg_data, d, in_structure = get_fgbuster_data
    nu = fg_data['frequencies']
    
    # Calculate Synchrotron with K_CMB unit in furax
    synch_fgbuster = fg_data['SYNC_K_CMB'][..., jnp.newaxis, jnp.newaxis] * fg_data['freq_maps']
    synch_fgbuster_tree = StokesPyTree.from_stokes(I=synch_fgbuster[:, 0, :],
                                                   Q=synch_fgbuster[:, 1, :],
                                                   U=synch_fgbuster[:, 2, :])
    
    synch_operator = SynchrotronOperator(nu, in_structure=in_structure, frequency0=20., units='K_CMB', beta_pl=-3.0)
    synch_furax = synch_operator(d)
    
    assert jax.tree.all(jax.tree.map(jnp.allclose, synch_furax, synch_fgbuster_tree))


def test_synchrotron_k_rj(get_fgbuster_data):
    fg_data, d, in_structure = get_fgbuster_data
    nu = fg_data['frequencies']
    
    # Calculate Synchrotron with K_RJ unit in furax
    synch_fgbuster = fg_data['SYNC_K_RJ'][..., jnp.newaxis, jnp.newaxis] * fg_data['freq_maps']
    synch_fgbuster_tree = StokesPyTree.from_stokes(I=synch_fgbuster[:, 0, :],
                                                   Q=synch_fgbuster[:, 1, :],
                                                   U=synch_fgbuster[:, 2, :])
    
    synch_operator = SynchrotronOperator(nu, in_structure=in_structure, frequency0=20., units='K_RJ', beta_pl=-3.0)
    synch_furax = synch_operator(d)
    
    assert jax.tree.all(jax.tree.map(jnp.allclose, synch_furax, synch_fgbuster_tree))
