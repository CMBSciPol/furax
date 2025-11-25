
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

ComponentParametersDict = dict[str, Stokes]

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
    """
    Create a linear operator component corresponding to the given astrophysical signal.

    Parameters
    ----------
    name : str
        Name of the component ('cmb', 'dust', or 'synchrotron').
    nu : Array
        Array of frequencies at which the operator is evaluated.
    frequency0 : float
        Reference frequency. For dust, this is dust_nu0; for synchrotron, synchrotron_nu0.
    params : PyTree[Array]
        Dictionary containing the spectral parameters, e.g., 'temp_dust', 'beta_dust', or 'beta_pl'.
    patch_indices : PyTree[Array]
        Dictionary containing the patch indices for spatially varying parameters.
    in_structure : Stokes
        The input structure (e.g., a Stokes object) defining the shape and configuration.

    Returns
    -------
    AbstractLinearOperator
        The corresponding linear operator for the specified component.

    Raises
    ------
    ValueError
        If the component name is not one of 'cmb', 'dust', or 'synchrotron'.
    """
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
    """
    Determine the list of available astrophysical components based on the provided parameters.

    Parameters
    ----------
    params : PyTree[Array]
        Dictionary containing spectral parameters. Expected keys include 'temp_dust', 'beta_dust',
        and/or 'beta_pl'.

    Returns
    -------
    list[str]
        List of available components. 'cmb' is always included; 'dust' is added if both
        'temp_dust' and 'beta_dust' are provided; 'synchrotron' is added if 'beta_pl' is provided.

    Raises
    ------
    AssertionError
        If only one of 'temp_dust' or 'beta_dust' is provided without the other.
    """
    available_components = ['cmb']
    if 'temp_dust' in params or 'beta_dust' in params:
        assert 'temp_dust' in params and 'beta_dust' in params, (
            'Both temp_dust and beta_dust must be provided'
        )
        available_components.append('dust')
    if 'beta_pl' in params:
        available_components.append('synchrotron')
    return available_components


# Remove JIT for now - can be added back once we ensure all static args are truly static
def _spectral_likelihood_core_bma(
    params: PyTree[Array],
    patch_indices: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    BMA: AbstractLinearOperator,
    N_2: AbstractLinearOperator | None = None,
) -> tuple[ComponentParametersDict, ComponentParametersDict]:
    """
    Compute the base spectral log likelihood components using the BMA operator.

    This function computes two key quantities:
    - BMA_T_N_inv_d: The product BMA^T N^{-1} d
    - s: Sky vector by (BMA^T N^{-1} BMA)^{-1} (BMA^T N^{-1} d)

    Mathematically, this corresponds to:

    $$
    \\left(BMA^T N^{-1} d\\right)^T \\left(BMA^T N^{-1} BMA\\right)^{-1} \\left(BMA^T N^{-1} d\\right)
    $$

    where:
      - $BMA$ is the complete operator (Bandpass @ HWP @ Mixing matrix @ Additional operators).
      - $N$ is the noise operator.
      - $d$ is the observed data in Stokes parameters.

    Parameters
    ----------
    params : PyTree[Array]
        Dictionary of spectral parameters.
    patch_indices : PyTree[Array]
        Dictionary of patch indices for spatially varying parameters.
    nu : Array
        Array of frequencies.
    N : AbstractLinearOperator
        Noise covariance operator.
    d : Stokes
        Data in Stokes parameter format.
    dust_nu0 : float
        Reference frequency for dust.
    synchrotron_nu0 : float
        Reference frequency for synchrotron.
    BMA : AbstractLinearOperator
        The complete BMA operator (includes mixing matrix, HWP, bandpass, etc.).
    N_2 : AbstractLinearOperator or None
        Optional secondary noise operator; if None, it defaults to N.

    Returns
    -------
    tuple[SpecParamType, SpecParamType]
        A tuple containing:
          - BMA_T_N_inv_d: The weighted data vector BMA^T N^{-1} d.
          - s: The solution vector (BMA^T N^{-1} BMA)^{-1} (BMA^T N^{-1} d).

    Raises
    ------
    AssertionError
        If provided keys in params or patch_indices are not within the valid sets.
    """
    if N_2 is None:
        N_2 = N

    assert set(params.keys()).issubset(valid_keys), (
        f'params.keys(): {params.keys()} , valid_keys: {valid_keys}'
    )
    assert set(patch_indices.keys()).issubset(valid_patch_keys), (
        f'patch_indices.keys(): {patch_indices.keys()} , valid_patch_keys: {valid_patch_keys}'
    )

    BMA_T_N_inv_d = (BMA.T @ N.I)(d)
    s = (BMA.T @ N_2.I @ BMA).I(BMA_T_N_inv_d)

    return BMA_T_N_inv_d, s


# Remove JIT for now
def sky_signal_bma(
    params: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    BMA: AbstractLinearOperator,
    patch_indices: PyTree[Array] = single_cluster_indices,
    N_2: AbstractLinearOperator | None = None,
) -> ComponentParametersDict:
    """
    Compute the estimated sky signal based on the BMA operator.

    This function extracts the sky vector 's' from the base spectral log likelihood
    computation using the BMA operator, which represents the reconstructed sky signal.

    Parameters
    ----------
    params : PyTree[Array]
        Dictionary of spectral parameters.
    nu : Array
        Array of frequencies.
    N : AbstractLinearOperator
        Noise covariance operator.
    d : Stokes
        Data in Stokes parameters.
    dust_nu0 : float
        Reference frequency for dust.
    synchrotron_nu0 : float
        Reference frequency for synchrotron.
    BMA : AbstractLinearOperator
        The complete BMA operator.
    patch_indices : PyTree[Array], optional
        Patch indices for spatially varying parameters (default is single_cluster_indices).
    N_2 : AbstractLinearOperator or None
        Optional secondary noise operator; if None, it defaults to N.

    Returns
    -------
    ComponentParametersDict
        Estimated sky signal for each component.
    """
    _, s = _spectral_likelihood_core_bma(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0, BMA, N_2
    )
    return cast(ComponentParametersDict, s)


# Remove JIT for now
def spectral_log_likelihood_bma(
    params: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    BMA: AbstractLinearOperator,
    patch_indices: PyTree[Array] = single_cluster_indices,
    N_2: AbstractLinearOperator | None = None,
) -> Scalar:
    """
    Compute the spectral log likelihood using the BMA operator.

    The likelihood is calculated based on the weighted data vector and its associated solution,
    using the complete BMA operator instead of just the mixing matrix.

    Parameters
    ----------
    params : PyTree[Array]
        Dictionary of spectral parameters.
    nu : Array
        Array of frequencies.
    N : AbstractLinearOperator
        Noise covariance operator.
    d : Stokes
        Data in Stokes parameters.
    dust_nu0 : float
        Reference frequency for dust.
    synchrotron_nu0 : float
        Reference frequency for synchrotron.
    BMA : AbstractLinearOperator
        The complete BMA operator.
    patch_indices : PyTree[Array], optional
        Patch indices for spatially varying parameters (default is single_cluster_indices).
    N_2 : AbstractLinearOperator or None
        Optional secondary noise operator; if None, it defaults to N.

    Returns
    -------
    Scalar
        The spectral log likelihood value.
    """
    BMA_T_N_inv_d, s = _spectral_likelihood_core_bma(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0, BMA, N_2
    )
    ll: Scalar = dot(BMA_T_N_inv_d, s)
    return ll


# Remove JIT for now
def negative_log_likelihood_bma(
    params: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    BMA: AbstractLinearOperator,
    patch_indices: PyTree[Array] = single_cluster_indices,
    N_2: AbstractLinearOperator | None = None,
) -> Scalar:
    """
    Compute the negative spectral log likelihood using the BMA operator.

    This function returns the negative of the spectral log likelihood using the complete
    BMA operator, which is useful for optimization procedures where minimizing the negative 
    log likelihood is equivalent to maximizing the likelihood.

    Parameters
    ----------
    params : PyTree[Array]
        Dictionary of spectral parameters.
    nu : Array
        Array of frequencies.
    N : AbstractLinearOperator
        Noise covariance operator.
    d : Stokes
        Data in Stokes parameters.
    dust_nu0 : float
        Reference frequency for dust.
    synchrotron_nu0 : float
        Reference frequency for synchrotron.
    BMA : AbstractLinearOperator
        The complete BMA operator.
    patch_indices : PyTree[Array], optional
        Patch indices for spatially varying parameters (default is single_cluster_indices).
    N_2 : AbstractLinearOperator or None
        Optional secondary noise operator; if None, it defaults to N.

    Returns
    -------
    Scalar
        The negative spectral log likelihood.
    """
    nll: Scalar = -spectral_log_likelihood_bma(
        params, nu, N, d, dust_nu0, synchrotron_nu0, BMA, patch_indices, N_2
    )
    return nll

@partial(jax.jit, static_argnums=(4, 5))
def sky_signal(
    params: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    BMA: AbstractLinearOperator,
    patch_indices: PyTree[Array] = single_cluster_indices,
    op: AbstractLinearOperator | None = None,
    N_2: AbstractLinearOperator | None = None,
) -> ComponentParametersDict:
    """
    Compute the estimated sky signal based on the provided spectral parameters.

    This function extracts the sky vector 's' from the base spectral log likelihood
    computation, which represents the reconstructed sky signal.

    Parameters
    ----------
    params : PyTree[Array]
        Dictionary of spectral parameters.
    nu : Array
        Array of frequencies.
    N : AbstractLinearOperator
        Noise covariance operator.
    d : Stokes
        Data in Stokes parameters.
    dust_nu0 : float
        Reference frequency for dust.
    synchrotron_nu0 : float
        Reference frequency for synchrotron.
    patch_indices : PyTree[Array], optional
        Patch indices for spatially varying parameters (default is single_cluster_indices).

    Returns
    -------
    ComponentParametersDict
        Estimated sky signal for each component.
    """
    _, s = _spectral_likelihood_core_bma(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0, BMA, N_2
    )
    return cast(ComponentParametersDict, s)




