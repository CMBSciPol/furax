"""How many frozen atmosphere layers does the data hold? A data-driven count on simulated TOD.

Simulation: K_TRUE layers at different heights, each a von Karman screen (physical outer scale)
drifting with its own wind, read by `FourierScreenOperator`, plus white noise.

On the screen coordinate u = (vx, vy) / vz a layer only shows its angular drift
omega = wind / height, so the data separate layers by omega. The heights enter only through the
angular scale of the turbulence (L0 / h), which the per-layer outer scale q0 = 2 pi h / L0 in u
captures as a nuisance parameter (not used to infer height).

Method (greedy, CLEAN-like):
  1. Scan omega on a grid: the marginal-likelihood gain of explaining the residual r with one
     more frozen screen at omega. (Plain chi^2 explained is biased to large |omega|: a faster
     drift spreads the samples over more of the box and frees more modes.)
  2. Add a layer at the peak, then jointly refine every (omega_k, A_k, q0_k) on the Gaussian
     marginal likelihood -2 ln L = d^T C^-1 d + ln det C, C = Phi S Phi^T + N.
  3. Stop when adding a layer no longer pays for its 4 parameters (BIC: 4 ln N_samp).
"""

import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.optimize import minimize
from screen_operator import FourierScreenOperator, half_plane_index, screen_coords

from furax.obs.coords import azel_to_quat, xieta_to_quat

jax.config.update('jax_enable_x64', True)

# --- truth: three layers ---------------------------------------------------------------------
HEIGHTS = np.array([500.0, 1500.0, 3000.0])  # m
WINDS = np.array([[2.0, 1.0], [-4.5, 4.5], [9.0, -9.0]])  # m/s
STDS = np.array([1.0, 0.6, 0.4])  # layer rms, arbitrary units
L0 = 300.0  # m, physical outer scale
# SIM_SPECTRUM=mismatched simulates slope -3 and L0 = 100 m; the fit keeps assuming von Karman
MISMATCHED = os.environ.get('SIM_SPECTRUM') == 'mismatched'
SIM_SLOPE, SIM_L0 = (-3.0, 100.0) if MISMATCHED else (-11 / 3, L0)
OMEGAS = WINDS / HEIGHTS[:, None]  # 1/s, the only kinematic quantity the data see
NOISE = 0.1  # white noise rms

# --- small CES observation -------------------------------------------------------------------
NDET, NSAMP, RATE = 8, 240, 2.4  # 100 s
times = jnp.arange(NSAMP) / RATE
qbore = azel_to_quat(jnp.deg2rad(10.0) * jnp.sin(2 * jnp.pi * times / 20.0), jnp.deg2rad(50.0))
key = jax.random.key(1)
k_det, k_atm, k_noise = jax.random.split(key, 3)
xi, eta = jnp.deg2rad(2.0) * jax.random.uniform(k_det, (2, NDET), minval=-1, maxval=1)
qdet = xieta_to_quat(xi, eta)
u = screen_coords(qbore, qdet)

# --- one Fourier box for every candidate omega -----------------------------------------------
OMEGA_MAX = 0.006  # search range, per axis
BOX = float(jnp.max(u.max((0, 1)) - u.min((0, 1)))) + OMEGA_MAX * float(times[-1]) + 0.2
index = half_plane_index(BOX, q_max=2 * np.pi / 0.1)
modes = 2 * np.pi / BOX * index
M = modes.shape[0]
q2 = jnp.sum(modes**2, -1)
print(f'box={BOX:.2f}, M={M} modes/layer ({2 * M} unknowns), samples={NDET * NSAMP}')


def unit_spectrum(q0, slope=-11 / 3):
    """Von Karman shape in u (or another slope), normalised to unit field variance."""
    s = (q2 + q0**2) ** (slope / 2)
    return jnp.concatenate([s, s]) / jnp.sum(s)


def design(omega):
    op = FourierScreenOperator.create(index, BOX, qbore, qdet, times, omega)
    return op, op.as_matrix()


# --- simulate --------------------------------------------------------------------------------
signal = jnp.zeros((NDET, NSAMP))
for k, (h, omega, std) in enumerate(zip(HEIGHTS, OMEGAS, STDS)):
    # physical q_phys = q_u / h, so the outer scale sits at q_u = 2 pi h / L0
    s = std**2 * unit_spectrum(2 * np.pi * h / SIM_L0, SIM_SLOPE)
    a = jnp.sqrt(s) * jax.random.normal(jax.random.fold_in(k_atm, k), (2 * M,))
    op, _ = design(jnp.asarray(omega))
    signal = signal + op(a.reshape(2, M))
d = (signal + NOISE * jax.random.normal(k_noise, signal.shape)).ravel()
print('true omegas [1e-3/s]:', np.round(OMEGAS * 1e3, 2).tolist())

# --- the fitted model: generic prior shape, free amplitude per layer --------------------------
Q0_FIT = 2 * np.pi / 0.3  # single outer scale in u for all layers (heights unknown)
NPAR = 4  # per layer: omega_x, omega_y, log amplitude, log outer-scale wavenumber in u


def neg2_log_like(params, d):
    """-2 ln L (no constant) for layers params = [(omega_x, omega_y, log A, log q0), ...]."""
    params = params.reshape(-1, NPAR)
    phi = jnp.concatenate([design(p[:2])[1] for p in params], axis=1)
    s = jnp.concatenate([jnp.exp(p[2]) * unit_spectrum(jnp.exp(p[3])) for p in params])
    ss = jnp.sqrt(s)
    g = ss[:, None] * (phi.T @ phi) * ss[None, :] / NOISE**2 + jnp.eye(s.size)
    b = ss * (phi.T @ d) / NOISE**2
    chol = jnp.linalg.cholesky(g)
    w = jax.scipy.linalg.solve_triangular(chol, b, lower=True)
    return d @ d / NOISE**2 - w @ w + 2 * jnp.sum(jnp.log(jnp.diag(chol)))


def scan_stat(omega, r, log_amp):
    """Marginal -2 dlnL gained by explaining the residual r with one more screen at omega."""
    return r @ r / NOISE**2 - neg2_log_like(jnp.array([*omega, log_amp, np.log(Q0_FIT)]), r)


def best_fit_signal(params):
    params = params.reshape(-1, NPAR)
    phi = jnp.concatenate([design(p[:2])[1] for p in params], axis=1)
    s = jnp.concatenate([jnp.exp(p[2]) * unit_spectrum(jnp.exp(p[3])) for p in params])
    a = jnp.linalg.solve(phi.T @ phi / NOISE**2 + jnp.diag(1 / s), phi.T @ d / NOISE**2)
    return phi @ a


grid = np.linspace(-OMEGA_MAX, OMEGA_MAX, 13)
gx, gy = np.meshgrid(grid, grid)
grid_omegas = jnp.stack([gx.ravel(), gy.ravel()], -1)
scan = jax.jit(
    lambda r, la: jax.lax.map(lambda w: scan_stat(w, r, la), grid_omegas).reshape(gx.shape)
)

bic = NPAR * np.log(d.size)
params = jnp.zeros((0, NPAR))
nll = float(d @ d / NOISE**2)  # K = 0
residual = d
scans, history = [], [(0, nll)]
for k in range(1, 6):
    t0 = time.time()
    # prior amplitude for the new screen: residual variance in excess of the noise
    log_amp = jnp.log(jnp.maximum(jnp.var(residual) - NOISE**2, NOISE**2))
    e = scan(residual, log_amp)
    scans.append(np.asarray(e))
    iy, ix = np.unravel_index(np.argmax(e), e.shape)
    start = jnp.concatenate(
        [params.ravel(), jnp.array([gx[iy, ix], gy[iy, ix], log_amp, np.log(Q0_FIT)])]
    )
    # omega = OMEGA_MAX tanh(z) keeps every drift inside the box, where the field does not wrap
    is_omega = jnp.tile(jnp.array([True, True, False, False]), k)

    def to_params(z, is_omega=is_omega):
        return jnp.where(is_omega, OMEGA_MAX * jnp.tanh(z), z)

    z0 = jnp.where(is_omega, jnp.arctanh(jnp.clip(start / OMEGA_MAX, -0.99, 0.99)), start)
    res = minimize(lambda z: neg2_log_like(to_params(z), d) / d.size, z0, method='BFGS')
    trial = to_params(res.x).reshape(-1, NPAR)
    trial_nll = float(neg2_log_like(trial.ravel(), d))
    gain = nll - trial_nll
    print(
        f'K={k}: scan peak at {np.round([gx[iy, ix] * 1e3, gy[iy, ix] * 1e3], 2).tolist()}, '
        f'-2 dlnL = {gain:.1f} (BIC cost {bic:.1f}), {time.time() - t0:.0f}s'
    )
    print(
        '     fitted omegas [1e-3/s]:',
        np.round(trial[:, :2] * 1e3, 2).tolist(),
        ' std:',
        np.round(np.exp(trial[:, 2] / 2), 2).tolist(),
    )
    history.append((k, trial_nll))
    if gain < bic:
        print(f'=> {k - 1} layers')
        break
    params, nll = trial, trial_nll
    residual = d - best_fit_signal(params.ravel())

# --- plot: one panel per scan, truths and fitted layers marked --------------------------------
fig, axes = plt.subplots(1, len(scans), figsize=(4 * len(scans), 4), squeeze=False)
ext = np.array([-OMEGA_MAX, OMEGA_MAX, -OMEGA_MAX, OMEGA_MAX]) * 1e3
for k, (ax, e) in enumerate(zip(axes[0], scans)):
    ax.imshow(e, origin='lower', extent=ext, cmap='viridis')
    ax.scatter(*(OMEGAS.T * 1e3), marker='x', c='r', label='true')
    if k:
        ax.scatter(
            *(np.asarray(params[:k, :2]).T * 1e3), marker='o', fc='none', ec='w', label='fitted'
        )
    ax.set_title(f'scan {k + 1}: residual after {k} layer(s)')
    ax.set_xlabel(r'$\omega_x$ [$10^{-3}$/s]')
axes[0, 0].set_ylabel(r'$\omega_y$ [$10^{-3}$/s]')
axes[0, 0].legend()
fig.tight_layout()
fig.savefig(__file__.replace('.py', '_mismatched.png' if MISMATCHED else '.png'))
