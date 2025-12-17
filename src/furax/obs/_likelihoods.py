import operator
from functools import partial
from typing import Any

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
) -> AbstractLinearOperator:  # Returns MixingMatrixOperator
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


@partial(jax.jit, static_argnums=(3, 4))
def preconditionner(
    params: PyTree[Array],
    nu: Array,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    patch_indices: PyTree[Array] = single_cluster_indices,
) -> MixingMatrixOperator:  # type: ignore[valid-type]
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


# ==============================================================================
# Custom VJP Implementation
# ==============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _spectral_log_likelihood_analytical(
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
    Forward pass wrapper. This is the main entry point.
    """
    AND, s = _spectral_likelihood_core(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0, op, N_2
    )
    ll: Scalar = dot(AND, s)
    return ll


def _spectral_log_likelihood_fwd(
    params: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    patch_indices: PyTree[Array],
    op: AbstractLinearOperator | None,
    N_2: AbstractLinearOperator | None,
) -> Any:
    """
    The forward pass implementation for custom_vjp.
    Returns the likelihood (L) and the residuals needed for the backward pass.
    """
    if N_2 is None:
        N_2 = N
    if op is None:
        op = IdentityOperator(d.structure)

    # Run the core logic to get 's' (sky signal)
    # We re-use the core function to ensure consistency
    AND, s = _spectral_likelihood_core(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0, op, N_2
    )

    L = dot(AND, s)

    # Save everything needed to reconstruct gradients
    # We save 's' directly. We save 'params' to reconstruct 'A'.
    res = (params, nu, N, d, s, op, N_2, dust_nu0, synchrotron_nu0, patch_indices)
    return L, res


def _spectral_log_likelihood_bwd(
    dust_nu0: float, synchrotron_nu0: float, res: Any, g: Scalar
) -> tuple[Any, ...]:
    """
    The backward pass implementation for custom_vjp.
    """
    (params, nu, N, d, s, op, N_2, _, _, patch_indices) = res

    # 1. Reconstruct the Mixing Matrix A from parameters
    in_structure = d.structure_for((d.shape[1],))
    A = _get_mixing_matrix(params, nu, dust_nu0, synchrotron_nu0, patch_indices, in_structure)

    # 2. Compute common terms
    # d_model = op * A * s
    d_model = op(A(s))

    # w_r is the generalized residual vector: N^{-1} d - N_2^{-1} d_model
    # If N == N_2, this is N^{-1} (d - d_model)
    term1 = N.I(d)
    term2 = N_2.I(d_model)
    w_r = term1 - term2

    # -----------------------------------------------------------
    # Gradient w.r.t Data (d)
    # dL/dd = 2 * N^{-1} * (op * A * s) = 2 * N^{-1} * d_model
    # Multiplied by incoming gradient 'g'
    # -----------------------------------------------------------
    # We use term2 (N_2.I(d_model)) if N==N_2, but strictly it is N.I(d_model) for dL/dd.
    # If N != N_2, the likelihood definition is slightly ambiguous without explicit math,
    # but based on L = d.T N.I A s, the derivative is 2 N.I A s.
    d_grad = (2 * g) * N.I(d_model)

    # -----------------------------------------------------------
    # Gradient w.r.t Params
    # dL/dA = 2 * op.T * w_r * s.T (outer product)
    # We compute vjp of (params -> A(s)) against vector (2 * op.T * w_r)
    # -----------------------------------------------------------

    # The 'cotangent' vector for the VJP
    u_vec = (2 * g) * op.T(w_r)

    def apply_A_to_fixed_s(p: PyTree[Array]) -> Any:
        """Helper to differentiate A w.r.t p while holding s constant."""
        A_temp = _get_mixing_matrix(p, nu, dust_nu0, synchrotron_nu0, patch_indices, in_structure)
        return A_temp(s)

    # jax.vjp returns (primal_out, vjp_fun)
    _, vjp_fun = jax.vjp(apply_A_to_fixed_s, params)

    # Backpropagate u_vec to get gradients for params
    params_grad = vjp_fun(u_vec)[0]

    # Return gradients for differentiable inputs only: params and d
    # params (0), nu (1), N (2), d (3), patch_indices (6), op (7), N_2 (8)
    return (params_grad, None, None, d_grad, None, None, None)


# Register the custom VJP
_spectral_log_likelihood_analytical.defvjp(
    _spectral_log_likelihood_fwd, _spectral_log_likelihood_bwd
)


@partial(jax.jit, static_argnums=(4, 5, 9))
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
    analytical_gradient: bool = False,
) -> Scalar:
    """
    Compute the spectral log likelihood.

    Args:
        params (PyTree[Array]): Dictionary of spectral parameters.
        nu (Array): Array of frequencies.
        N (AbstractLinearOperator): Noise covariance operator.
        d (Stokes): Data in Stokes parameters.
        dust_nu0 (float): Reference frequency for dust.
        synchrotron_nu0 (float): Reference frequency for synchrotron.
        patch_indices (PyTree[Array], optional): Patch indices for spatially varying parameters (default is single_cluster_indices).
        op (AbstractLinearOperator or None, optional): Operator to be applied (default is None).
        N_2 (AbstractLinearOperator or None, optional): Secondary noise operator (default is None).
        analytical_gradient (bool, optional): If True, use the custom VJP implementation for analytical gradients.
                                            If False (default), use standard automatic differentiation.

    Returns:
        Scalar: The spectral log likelihood.

    Example:
        >>> from furax.obs import spectral_log_likelihood
        >>> from furax.obs.stokes import Stokes
        >>> from furax import HomothetyOperator
        >>> import jax.numpy as jnp
        >>> nside = 64
        >>> nu_freqs = jnp.array([30., 40., 100.])
        >>> d_data = Stokes.zeros((len(nu_freqs), 12 * nside**2)) # Example observed data
        >>> inv_noise = HomothetyOperator(jnp.ones(1), _in_structure=d_data.structure) # Example inverse noise
        >>> params = {'temp_dust': 20.0, 'beta_dust': 1.54, 'beta_pl': -3.0}
        >>> dust_nu0_ref = 150.0
        >>> synchrotron_nu0_ref = 20.0
        >>> ll_val = spectral_log_likelihood(params, nu_freqs, inv_noise, d_data, dust_nu0_ref, synchrotron_nu0_ref)
        >>> # print(ll_val)
    """
    if analytical_gradient:
        return _spectral_log_likelihood_analytical(
            params, nu, N, d, dust_nu0, synchrotron_nu0, patch_indices, op, N_2
        )

    AND, s = _spectral_likelihood_core(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0, op, N_2
    )
    ll: Scalar = dot(AND, s)
    return ll


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _sky_signal_analytical(
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
    Computes the estimated sky signal 's'.
    Wrapped with custom_vjp to handle the implicit differentiation of the linear solve.
    """
    _, s = _spectral_likelihood_core(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0, op, N_2
    )
    return s


def _sky_signal_fwd(
    params: PyTree[Array],
    nu: Array,
    N: AbstractLinearOperator,
    d: Stokes,
    dust_nu0: float,
    synchrotron_nu0: float,
    patch_indices: PyTree[Array],
    op: AbstractLinearOperator | None,
    N_2: AbstractLinearOperator | None,
) -> Any:
    if N_2 is None:
        N_2 = N
    if op is None:
        op = IdentityOperator(d.structure)

    # 1. Compute 's' using the core logic
    # We rely on the existing core function which builds A and solves M s = b
    _, s = _spectral_likelihood_core(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0, op, N_2
    )

    # 2. Save residuals for backward pass
    # We need 's' and all inputs to reconstruct the operators
    res = (params, nu, N, d, s, op, N_2, dust_nu0, synchrotron_nu0, patch_indices)

    return s, res


def _sky_signal_bwd(
    dust_nu0: float, synchrotron_nu0: float, res: Any, g: Scalar
) -> tuple[Any, ...]:
    """
    Backward pass for sky_signal.
    'g' is the incoming gradient (cotangent) w.r.t the output 's'.
    """
    (params, nu, N, d, s, op, N_2, _, _, patch_indices) = res

    # 1. Reconstruct Mixing Matrix A
    in_structure = d.structure_for((d.shape[1],))
    A = _get_mixing_matrix(params, nu, dust_nu0, synchrotron_nu0, patch_indices, in_structure)

    # 2. Reconstruct Curvature Operator M = A.T @ op.T @ N_2.I @ op @ A
    # (This is the same operator used to solve for s in the forward pass)
    M = A.T @ op.T @ N_2.I @ op @ A

    # 3. Solve the Adjoint System: M w = g
    # w represents how much the solution 's' shifts given the gradient 'g'
    w = M.I(g)

    # 4. Compute Gradient w.r.t Data (d)
    # d_bar = N.I @ op @ A @ w
    d_model_w = op(A(w))  # op @ A @ w
    d_grad = N.I(d_model_w)

    # 5. Compute Gradient w.r.t Params
    # We need two VJPs here corresponding to the two terms in the adjoint equation.

    # -- Term 1: Residual Push --
    # Vector u1 = op.T @ (N.I(d) - N_2.I(op(A(s))))
    d_model_s = op(A(s))
    residual_term = N.I(d) - N_2.I(d_model_s)
    u1 = op.T(residual_term)

    # Calculate VJP for: params -> A(w) against cotangent u1
    def apply_A_w(p: PyTree[Array]) -> Any:
        A_tmp = _get_mixing_matrix(p, nu, dust_nu0, synchrotron_nu0, patch_indices, in_structure)
        return A_tmp(w)  # Note: 'w' is fixed here

    _, vjp_A_w = jax.vjp(apply_A_w, params)
    grad_params_1 = vjp_A_w(u1)[0]

    # -- Term 2: Curvature Correction --
    # Vector u2 = - op.T @ N_2.I(op(A(w)))
    # Note: d_model_w was calculated in step 4
    u2 = -op.T(N_2.I(d_model_w))

    # Calculate VJP for: params -> A(s) against cotangent u2
    def apply_A_s(p: PyTree[Array]) -> Any:
        A_tmp = _get_mixing_matrix(p, nu, dust_nu0, synchrotron_nu0, patch_indices, in_structure)
        return A_tmp(s)  # Note: 's' is fixed here

    _, vjp_A_s = jax.vjp(apply_A_s, params)
    grad_params_2 = vjp_A_s(u2)[0]

    # Combine parameter gradients
    # params is a Pytree (dict), so we sum the gradients leaf-wise
    params_grad = jax.tree.map(lambda x, y: x + y, grad_params_1, grad_params_2)

    return (params_grad, None, None, d_grad, None, None, None)


# Register the custom VJP
_sky_signal_analytical.defvjp(_sky_signal_fwd, _sky_signal_bwd)


@partial(jax.jit, static_argnums=(4, 5, 9))
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
    analytical_gradient: bool = False,
) -> ComponentParametersDict:
    """
    Computes the estimated sky signal 's'.

    Args:
        params (PyTree[Array]): Dictionary of spectral parameters.
        nu (Array): Array of frequencies.
        N (AbstractLinearOperator): Noise covariance operator.
        d (Stokes): Data in Stokes parameters.
        dust_nu0 (float): Reference frequency for dust.
        synchrotron_nu0 (float): Reference frequency for synchrotron.
        patch_indices (PyTree[Array], optional): Patch indices for spatially varying parameters (default is single_cluster_indices).
        op (AbstractLinearOperator or None, optional): Operator to be applied (default is None).
        N_2 (AbstractLinearOperator or None, optional): Secondary noise operator (default is None).
        analytical_gradient (bool, optional): If True, use the custom VJP implementation for analytical gradients.
                                            If False (default), use standard automatic differentiation.

    Returns:
        ComponentParametersDict: The estimated sky signal components (e.g., 'cmb', 'dust', 'synchrotron').

    Example:
        >>> from furax.obs import sky_signal
        >>> from furax.obs.stokes import Stokes
        >>> from furax import HomothetyOperator
        >>> import jax.numpy as jnp
        >>> nside = 64
        >>> nu_freqs = jnp.array([30., 40., 100.])
        >>> d_data = Stokes.zeros((len(nu_freqs), 12 * nside**2)) # Example observed data
        >>> inv_noise = HomothetyOperator(jnp.ones(1), _in_structure=d_data.structure) # Example inverse noise
        >>> params = {'temp_dust': 20.0, 'beta_dust': 1.54, 'beta_pl': -3.0}
        >>> dust_nu0_ref = 150.0
        >>> synchrotron_nu0_ref = 20.0
        >>> sky_comp = sky_signal(params, nu_freqs, inv_noise, d_data, dust_nu0_ref, synchrotron_nu0_ref)
        >>> # print(sky_comp['cmb'].i.shape)
    """
    if analytical_gradient:
        return _sky_signal_analytical(
            params, nu, N, d, dust_nu0, synchrotron_nu0, patch_indices, op, N_2
        )

    _, s = _spectral_likelihood_core(
        params, patch_indices, nu, N, d, dust_nu0, synchrotron_nu0, op, N_2
    )
    return s


@partial(jax.jit, static_argnums=(4, 5, 9))
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
    analytical_gradient: bool = False,
) -> Scalar:
    """
    Compute the negative spectral log likelihood.

    This function returns the negative of the spectral log likelihood, which is useful for
    optimization procedures where minimizing the negative log likelihood is equivalent to
    maximizing the likelihood.

    Args:
        params (PyTree[Array]): Dictionary of spectral parameters.
        nu (Array): Array of frequencies.
        N (AbstractLinearOperator): Noise covariance operator.
        d (Stokes): Data in Stokes parameters.
        dust_nu0 (float): Reference frequency for dust.
        synchrotron_nu0 (float): Reference frequency for synchrotron.
        patch_indices (PyTree[Array], optional): Patch indices for spatially varying parameters (default is single_cluster_indices).
        op (AbstractLinearOperator or None, optional): Operator to be applied (default is None).
        N_2 (AbstractLinearOperator or None, optional): Secondary noise operator (default is None).
        analytical_gradient (bool, optional): If True, use the custom VJP implementation for analytical gradients.
                                            If False (default), use standard automatic differentiation.

    Returns:
        Scalar: The negative spectral log likelihood.

    Example:
        >>> from furax.obs import negative_log_likelihood
        >>> from furax.obs.stokes import Stokes
        >>> from furax import HomothetyOperator
        >>> import jax.numpy as jnp
        >>> nside = 64
        >>> nu_freqs = jnp.array([30., 40., 100.])
        >>> d_data = Stokes.zeros((len(nu_freqs), 12 * nside**2)) # Example observed data
        >>> inv_noise = HomothetyOperator(jnp.ones(1), _in_structure=d_data.structure) # Example inverse noise
        >>> params = {'temp_dust': 20.0, 'beta_dust': 1.54, 'beta_pl': -3.0}
        >>> dust_nu0_ref = 150.0
        >>> synchrotron_nu0_ref = 20.0
        >>> nll_val = negative_log_likelihood(params, nu_freqs, inv_noise, d_data, dust_nu0_ref, synchrotron_nu0_ref)
        >>> # print(nll_val)
    """
    nll: Scalar = -spectral_log_likelihood(
        params,
        nu,
        N,
        d,
        dust_nu0,
        synchrotron_nu0,
        patch_indices,
        op,
        N_2,
        analytical_gradient=analytical_gradient,
    )
    return nll


@partial(jax.jit, static_argnums=(4, 5, 9))
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
    analytical_gradient: bool = False,
) -> Scalar:
    """
    Compute the variance of the CMB component from the spectral estimation.

    This function calculates the variance of the CMB component from the estimated sky signal 's'.

    Args:
        params (PyTree[Array]): Dictionary of spectral parameters.
        nu (Array): Array of frequencies.
        N (AbstractLinearOperator): Noise covariance operator.
        d (Stokes): Data in Stokes parameters.
        dust_nu0 (float): Reference frequency for dust.
        synchrotron_nu0 (float): Reference frequency for synchrotron.
        patch_indices (PyTree[Array], optional): Patch indices for spatially varying parameters (default is single_cluster_indices).
        op (AbstractLinearOperator or None, optional): Operator to be applied (default is None).
        N_2 (AbstractLinearOperator or None, optional): Secondary noise operator (default is None).
        analytical_gradient (bool, optional): If True, use the custom VJP implementation for analytical gradients.
                                            If False (default), use standard automatic differentiation.

    Returns:
        Scalar: The variance of the CMB component.

    Example:
        >>> from furax.obs import spectral_cmb_variance
        >>> from furax.obs.stokes import Stokes
        >>> from furax import HomothetyOperator
        >>> import jax.numpy as jnp
        >>> nside = 64
        >>> nu_freqs = jnp.array([30., 40., 100.])
        >>> d_data = Stokes.zeros((len(nu_freqs), 12 * nside**2)) # Example observed data
        >>> inv_noise = HomothetyOperator(jnp.ones(1), _in_structure=d_data.structure) # Example inverse noise
        >>> params = {'temp_dust': 20.0, 'beta_dust': 1.54, 'beta_pl': -3.0}
        >>> dust_nu0_ref = 150.0
        >>> synchrotron_nu0_ref = 20.0
        >>> cmb_var_val = spectral_cmb_variance(params, nu_freqs, inv_noise, d_data, dust_nu0_ref, synchrotron_nu0_ref)
        >>> # print(cmb_var_val)
    """
    s = sky_signal(
        params,
        nu,
        N,
        d,
        dust_nu0,
        synchrotron_nu0,
        patch_indices,
        op,
        N_2,
        analytical_gradient=analytical_gradient,
    )

    cmb_var: Scalar = jax.tree.reduce(operator.add, jax.tree.map(jnp.var, s['cmb']))
    return cmb_var
