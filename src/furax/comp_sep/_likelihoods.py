import operator
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Scalar

from furax import AbstractLinearOperator, IdentityOperator
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
valid_keys = {'temp_dust', 'beta_dust', 'beta_pl'}
valid_patch_keys = {'temp_dust_patches', 'beta_dust_patches', 'beta_pl_patches'}


def _create_component(
    name: str,
    nu: Array,
    frequency0: float,
    params: PyTree[Array],
    patch_indices: PyTree[Array],
    in_structure: Stokes,
) -> AbstractLinearOperator:
    if name == 'cmb':
        return CMBOperator(nu, in_structure=in_structure)
    elif name == 'dust':
        return DustOperator(
            nu,
            frequency0=frequency0,
            temperature=params['temp_dust'],
            temperature_patch_indices=patch_indices['temp_dust_patches'],
            beta=params['beta_dust'],
            beta_patch_indices=patch_indices['beta_dust_patches'],
            in_structure=in_structure,
        )
    elif name == 'synchrotron':
        return SynchrotronOperator(
            nu,
            frequency0=frequency0,
            beta_pl=params['beta_pl'],
            beta_pl_patch_indices=patch_indices['beta_pl_patches'],
            in_structure=in_structure,
        )
    else:
        raise ValueError(f'Unknown component: {name}')


def _get_available_components(params: PyTree[Array]) -> list[str]:
    available_components = ['cmb']
    if 'temp_dust' in params or 'beta_dust' in params:
        assert 'temp_dust' in params and 'beta_dust' in params, (
            'Both temp_dust and beta_dust must be provided'
        )
        available_components.append('dust')
    if 'beta_pl' in params:
        available_components.append('synchrotron')
    return available_components


@partial(jax.jit, static_argnums=(5, 6))
def _base_spectral_log_likelihood(
    params: PyTree[Array],
    patch_indices: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    op: AbstractLinearOperator | None,
    N_2: AbstractLinearOperator | None,
) -> tuple[SpecParamType, SpecParamType]:
    in_structure = d.structure_for((d.shape[1],))

    if N_2 is None:
        N_2 = N

    if op is None:
        op = IdentityOperator(d.structure)

    assert set(params.keys()).issubset(valid_keys), (
        f'params.keys(): {params.keys()} , valid_keys: {valid_keys}'
    )
    assert set(patch_indices.keys()).issubset(valid_patch_keys), (
        f'patch_indices.keys(): {patch_indices.keys()} , valid_patch_keys: {valid_patch_keys}'
    )

    components = {}
    for component in _get_available_components(params):
        components[component] = _create_component(
            component,
            nu,
            dust_nu0 if component == 'dust' else synchrotron_nu0,
            params,
            patch_indices,
            in_structure,
        )

    A = MixingMatrixOperator(**components)

    AND = (A.T @ op.T @ N.I)(d)
    s = (A.T @ op.T @ N_2.I @ op @ A).I(AND)

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
