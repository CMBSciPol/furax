from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float
from optax import tree_utils as otu
from scipy.signal import get_window
from toast import qarray as qa


# TODO jax version
def get_local_meridian_angle(quat):  # type: ignore[no-untyped-def]
    """
    Compute angle between local meridian and orientation vector from quaternions.

    Assumes that the quaternions encode the rotation between the celestial frame
    and some other frame (e.g. detector or boresight frame). The "orientation vector"
    is the unit vector of the latter frame obtained by rotating the X axis of the
    celestial frame. For a detector this will be the polarization sensitive direction.
    The local meridian vector is obtained by projecting the -Z axis of the celestial
    frame onto the plane orthogonal to the pointing direction.

    taken from
    https://github.com/hpc4cmb/toast/blob/toast3/src/toast/ops/stokes_weights/kernels_numpy.py#L12
    """
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    xaxis = np.array([1, 0, 0], dtype=np.float64)

    vd = qa.rotate(quat, zaxis)
    vo = qa.rotate(quat, xaxis)

    # The vector orthogonal to the line of sight that is parallel
    # to the local meridian.
    dir_ang = np.arctan2(vd[:, 1], vd[:, 0])
    dir_r = np.sqrt(1.0 - vd[:, 2] * vd[:, 2])
    vm_z = -dir_r
    vm_x = vd[:, 2] * np.cos(dir_ang)
    vm_y = vd[:, 2] * np.sin(dir_ang)

    # Compute the rotation angle from the meridian vector to the
    # orientation vector.  The direction vector is normal to the plane
    # containing these two vectors, so the rotation angle is:
    #
    # angle = atan2((v_m x v_o) . v_d, v_m . v_o)
    #
    alpha_y = (
        vd[:, 0] * (vm_y * vo[:, 2] - vm_z * vo[:, 1])
        - vd[:, 1] * (vm_x * vo[:, 2] - vm_z * vo[:, 0])
        + vd[:, 2] * (vm_x * vo[:, 1] - vm_y * vo[:, 0])
    )
    alpha_x = vm_x * vo[:, 0] + vm_y * vo[:, 1] + vm_z * vo[:, 2]

    alpha = np.arctan2(alpha_y, alpha_x)
    return alpha


def next_fast_fft_size(n: int) -> int:
    return int(2 ** np.ceil(np.log2(n)))


@partial(jax.jit, static_argnames=['fft_size'])
def interpolate_psd(
    freq: Float[Array, 'a b'],
    psd: Float[Array, 'a b'],
    fft_size: int,
    rate: float = 1.0,
) -> Float[Array, 'a {fft_size // 2 + 1}']:
    """Perform a logarithmic interpolation of PSD values."""
    interp_freq = jnp.fft.rfftfreq(fft_size, 1 / rate)
    # shift by fixed amounts in frequency and amplitude to avoid zeros
    freq_shift = rate / fft_size
    psd_shift = 0.01 * jnp.min(jnp.where(psd > 0, psd, 0))
    log_x = jnp.log10(interp_freq + freq_shift)
    log_xp = jnp.log10(freq + freq_shift)
    log_fp = jnp.log10(psd + psd_shift)
    # vectorize jax.numpy.interp, which only works on one-dimensional arrays
    f = partial(jnp.interp, left='extrapolate', right='extrapolate')
    vf = jnp.vectorize(f, signature='(m),(n),(n)->(m)')
    interp_psd: Array = vf(log_x, log_xp, log_fp)
    interp_psd = jnp.power(10.0, interp_psd) - psd_shift
    # zero out DC value
    interp_psd.at[..., 0].set(0)
    return interp_psd


@partial(jax.jit, static_argnames=['fft_size', 'rate'])
def estimate_psd(
    tod: Float[Array, 'a b'], fft_size: int, rate: float
) -> Float[Array, 'a {fft_size // 2 + 1}']:
    # average the periodogram estimate over 10 minute segments
    nperseg = int(600 * rate)
    f, Pxx = jax.scipy.signal.welch(tod, fs=rate, nperseg=nperseg)
    # fit and compute full size PSD from fitted parameters
    params = fit_psd_model(f, Pxx)
    freq = jnp.fft.rfftfreq(fft_size, 1 / rate)
    func = jnp.vectorize(_model, signature='(p),(n)->(n)')
    psd: Array = func(params, freq)
    return psd


@partial(jnp.vectorize, signature='(n),(n)->(p)')
def fit_psd_model(f: Float[Array, ' a'], Pxx: Float[Array, ' a']) -> Float[Array, '4']:
    """Fit a 1/f PSD model to the periodogram in log space."""
    loss = partial(_compute_loss, x=f[1:], y=jnp.log10(Pxx[1:]))
    opt = optax.lbfgs()
    init_params = jnp.array([1.0, -1.0, 0.1, 1e-5])
    final_params, _ = run_lbfgs(init_params, loss, opt, max_iter=100, tol=1e-3)
    return final_params


def _compute_loss(
    params: Float[Array, '4'], x: Float[Array, ' a'], y: Float[Array, ' a']
) -> Float[Array, '1']:
    y_pred = _log_model(params, x)
    loss = jnp.mean(optax.l2_loss(y_pred, y))
    return loss


def _log_model(params: Float[Array, '4'], x: Float[Array, ' a']) -> Float[Array, ' a']:
    sigma, alpha, fk, f0 = _unpack(params)
    log_psd = 2 * jnp.log10(sigma) + jnp.log10(1 + ((x + f0) / fk) ** alpha)
    return log_psd


def _model(params: Float[Array, '4'], x: Float[Array, ' a']) -> Float[Array, ' a']:
    sigma, alpha, fk, f0 = _unpack(params)
    psd = sigma**2 * (1 + ((x + f0) / fk) ** alpha)
    return psd


def _unpack(params: Float[Array, '4']) -> tuple[Array, Array, Array, Array]:
    net = params[0]
    alpha = params[1]
    f_knee = params[2]
    f_min = params[3]
    return net, alpha, f_knee, f_min


def run_lbfgs(
    init_params: Array, fun: Any, opt: Any, max_iter: int, tol: float
) -> tuple[Array, Any]:
    """Minimizes a function using the L-BFGS solver.

    From https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html#l-bfgs-solver
    """
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):  # type: ignore[no-untyped-def]
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=fun)
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):  # type: ignore[no-untyped-def]
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(continuing_criterion, step, init_carry)
    return final_params, final_state


@partial(jax.jit, static_argnames=['correlation_length'])
def psd_to_invntt(
    psd: Float[Array, '...'], correlation_length: int
) -> Float[Array, '... {correlation_length}']:
    """Compute the inverse autocorrelation function from PSD values.
    The result is apodized and cut at the specified correlation length.
    """
    invntt = jnp.fft.irfft(1 / psd)[..., :correlation_length]
    window = apodization_window(correlation_length)
    return invntt * window


def apodization_window(size: int, kind: str = 'chebwin') -> Float[Array, ' {size}']:
    window_type: tuple[Any, ...]
    if kind == 'gaussian':
        q_apo = 3  # apodization factor: cut happens at q_apo * sigma in the Gaussian window
        window_type = ('general_gaussian', 1, 1 / q_apo * size)
    elif kind == 'chebwin':
        at = 150  # attenuation level (dB)
        window_type = ('chebwin', at)
    else:
        raise RuntimeError(f'Apodization window {kind!r} is not supported.')

    window = jnp.array(get_window(window_type, 2 * size))
    window = jnp.fft.ifftshift(window)[:size]
    return window