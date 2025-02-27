from functools import partial
import jax
import jax.numpy as jnp

from furax.obs.operators._seds import CMBOperator, DustOperator, SynchrotronOperator, MixingMatrixOperator
import operator

from furax.tree import dot

single_cluster_indices = patch_indices = {
    'temp_dust_patches': None,
    'beta_dust_patches': None,
    'beta_pl_patches': None,
}


@partial(jax.jit, static_argnums=(5, 6))
def _base_spectral_log_likelihood(params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0):
    in_structure = d.structure_for((d.shape[1],))

    cmb = CMBOperator(nu, in_structure=in_structure)
    dust = DustOperator(
        nu,
        frequency0=dust_nu0,
        temperature=params['temp_dust'],
        temperature_patch_indices=patch_indices['temp_dust_patches'],
        beta=params['beta_dust'],
        beta_patch_indices=patch_indices['beta_dust_patches'],
        in_structure=in_structure,
    )
    synchrotron = SynchrotronOperator(
        nu,
        frequency0=synchrotron_nu0,
        beta_pl=params['beta_pl'],
        beta_pl_patch_indices=patch_indices['beta_pl_patches'],
        in_structure=in_structure,
    )

    A = MixingMatrixOperator(cmb=cmb, dust=dust, synchrotron=synchrotron)
    invN = N.I

    AND = (A.T @ invN)(d)
    s = (A.T @ invN @ A).I(AND)

    return AND, s


@partial(jax.jit, static_argnums=(4, 5))
def spectral_log_likelihood(
    params, nu, N, d, dust_nu0, synchrotron_nu0, patch_indices=single_cluster_indices
):
    AND, s = _base_spectral_log_likelihood(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0
    )
    return dot(AND, s) 


@partial(jax.jit, static_argnums=(4, 5))
def negative_log_likelihood(
    params, nu, N, d, dust_nu0, synchrotron_nu0, patch_indices=single_cluster_indices
):
    return -spectral_log_likelihood(params, nu, N, d, dust_nu0, synchrotron_nu0, patch_indices)


@partial(jax.jit, static_argnums=(4, 5))
def spectral_cmb_variance(
    params, nu, N, d, dust_nu0, synchrotron_nu0, patch_indices=single_cluster_indices
):
    _, s = _base_spectral_log_likelihood(params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0)
    return jax.tree.reduce(operator.add, jax.tree.map(jnp.var, s['cmb']))
