import operator
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Scalar

from furax import AbstractLinearOperator
from furax.obs.operators._seds import (
    CMBOperator,
    DustOperator,
    MixingMatrixOperator,
    SynchrotronOperator,
)
from furax.obs.stokes import Stokes
from furax.tree import dot

SpecParamType = dict[str, Stokes]

single_cluster_indices = {
    'temp_dust_patches': None,
    'beta_dust_patches': None,
    'beta_pl_patches': None,
}


@partial(jax.jit, static_argnums=(5, 6))
def _base_spectral_log_likelihood(
    params: PyTree[Array],
    patch_indices: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
) -> tuple[SpecParamType, SpecParamType]:
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
def sky_signal(
    params: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    patch_indices: PyTree[Array] = single_cluster_indices,
) -> SpecParamType:
    _, s = _base_spectral_log_likelihood(params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0)
    return cast(SpecParamType, s)


@partial(jax.jit, static_argnums=(4, 5))
def spectral_log_likelihood(
    params: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    patch_indices: PyTree[Array] = single_cluster_indices,
) -> Scalar:
    AND, s = _base_spectral_log_likelihood(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0
    )
    ll: Scalar = dot(AND, s)
    return ll


@partial(jax.jit, static_argnums=(4, 5))
def negative_log_likelihood(
    params: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    patch_indices: PyTree[Array] = single_cluster_indices,
) -> Scalar:
    nll: Scalar = -spectral_log_likelihood(
        params, nu, N, d, dust_nu0, synchrotron_nu0, patch_indices
    )
    return nll


@partial(jax.jit, static_argnums=(4, 5))
def spectral_cmb_variance(
    params: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    patch_indices: PyTree[Array] = single_cluster_indices,
) -> Scalar:
    _, s = _base_spectral_log_likelihood(params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0)
    cmb_var: Scalar = jax.tree.reduce(operator.add, jax.tree.map(jnp.var, s['cmb']))
    return cmb_var
