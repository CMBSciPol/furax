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


def _get_mixing_matrix(
    params: PyTree[Array],
    nu: Array,
    dust_nu0: float,
    synchrotron_nu0: float,
    patch_indices: PyTree[Array],
    in_structure: Stokes,
) -> MixingMatrixOperator:
    """Helper to construct the MixingMatrixOperator from parameters."""
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
    return MixingMatrixOperator(**components)


@jax.jit(static_argnums=(3, 4))
def preconditionner(
    params: PyTree[Array],
    nu: Array,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    patch_indices: PyTree[Array] = single_cluster_indices,
) -> MixingMatrixOperator:
    """
    Constructs the MixingMatrixOperator for preconditioning purposes.

    This function builds the mixing matrix operator based on the provided spectral parameters
    and frequencies, without directly involving the observed data or noise operators.
    It is typically used to create a preconditioner for iterative solvers in component separation.

    Args:
        params (PyTree[Array]): Dictionary of spectral parameters.
        nu (Array): Array of frequencies.
        d (Stokes): Data in Stokes parameters, used only to infer the `in_structure`
                    for the MixingMatrixOperator. Its values are not used.
        dust_nu0 (float): Reference frequency for dust.
        synchrotron_nu0 (float): Reference frequency for synchrotron.
        patch_indices (PyTree[Array], optional): Patch indices for spatially varying parameters (default is single_cluster_indices).

    Returns:
        MixingMatrixOperator: The constructed mixing matrix operator suitable for preconditioning.

    Example:
        >>> from furax.obs import preconditionner
        >>> from furax.obs.stokes import Stokes
        >>> import jax.numpy as jnp
        >>> nside = 64
        >>> nu_freqs = jnp.array([30., 40., 100.])
        >>> dummy_d = Stokes.zeros((len(nu_freqs), 12 * nside**2)) # Dummy data for structure
        >>> params = {'temp_dust': 20.0, 'beta_dust': 1.54, 'beta_pl': -3.0}
        >>> dust_nu0_ref = 150.0
        >>> synchrotron_nu0_ref = 20.0
        >>> A_precond = preconditionner(params, nu_freqs, dummy_d, dust_nu0_ref, synchrotron_nu0_ref)
        >>> # The preconditioner can now be used with a solver.
    """
    in_structure = Stokes.structure_for((d.shape[1],))
    A = _get_mixing_matrix(params, nu, dust_nu0, synchrotron_nu0, patch_indices, in_structure)
    return A


def _spectral_likelihood_core(
    params: PyTree[Array],
    patch_indices: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    op: AbstractLinearOperator | None,
    N_2: AbstractLinearOperator | None,
) -> tuple[ComponentParametersDict, ComponentParametersDict]:
    """
    Compute the base spectral log likelihood components used in spectral estimation.

    This function computes two key quantities:
    - AND: The product A^T N^{-1} d
    - s: Sky vector by (A^T N^{-1} A)^{-1} (A^T N^{-1} d)

    Mathematically, this corresponds to:

    $$
    \\left(A^T N^{-1} d\\right)^T \\left(A^T N^{-1} A\\right)^{-1} \\left(A^T N^{-1} d\\right)
    $$

    where:
      - $A$ is the mixing matrix operator constructed from the available components.
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
    op : AbstractLinearOperator or None
        Optional operator to be applied; if None, the IdentityOperator is used.
    N_2 : AbstractLinearOperator or None
        Optional secondary noise operator; if None, it defaults to N.

    Returns
    -------
    tuple[SpecParamType, SpecParamType]
        A tuple containing:
          - AND: The weighted data vector A^T N^{-1} d.
          - s: The solution vector (A^T N^{-1} A)^{-1} (A^T N^{-1} d).

    Raises
    ------
    AssertionError
        If provided keys in params or patch_indices are not within the valid sets.
    """
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

    # Use helper to build A
    A = _get_mixing_matrix(params, nu, dust_nu0, synchrotron_nu0, patch_indices, in_structure)

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
    _, s = _spectral_likelihood_core(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0, op, N_2
    )
    return cast(ComponentParametersDict, s)


@partial(jax.jit, static_argnums=(4, 5))
def spectral_log_likelihood(
    params: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    patch_indices: PyTree[Array] = single_cluster_indices,
    op: AbstractLinearOperator | None = None,
    N_2: AbstractLinearOperator | None = None,
) -> Scalar:
    """
    Compute the spectral log likelihood for the observed data.

    The likelihood is calculated based on the weighted data vector and its associated solution,
    as derived in the base spectral log likelihood.

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
    Scalar
        The spectral log likelihood value.
    """
    AND, s = _spectral_likelihood_core(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0, op, N_2
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
    op: AbstractLinearOperator | None = None,
    N_2: AbstractLinearOperator | None = None,
) -> Scalar:
    """
    Compute the negative spectral log likelihood.

    This function returns the negative of the spectral log likelihood, which is useful for
    optimization procedures where minimizing the negative log likelihood is equivalent to
    maximizing the likelihood.

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
    Scalar
        The negative spectral log likelihood.
    """
    nll: Scalar = -spectral_log_likelihood(
        params, nu, N, d, dust_nu0, synchrotron_nu0, patch_indices, op, N_2
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
    op: AbstractLinearOperator | None = None,
    N_2: AbstractLinearOperator | None = None,
) -> Scalar:
    """
    Compute the variance of the CMB component from the spectral estimation.

    This function calculates the variance of the CMB component by applying the base spectral log
    likelihood and then computing the variance over the resulting CMB signal.

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
    Scalar
        The variance of the CMB component.
    """
    _, s = _spectral_likelihood_core(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0, op, N_2
    )
    cmb_var: Scalar = jax.tree.reduce(operator.add, jax.tree.map(jnp.var, s['cmb']))
    return cmb_var
