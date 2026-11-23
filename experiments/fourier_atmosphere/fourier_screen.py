"""Prototype: a frozen atmosphere screen as a band-limited Fourier series, read by a furax operator.

One layer, white noise, dense non-uniform DFT (no NUFFT dependency). Checks:
  1. exact transpose,
  2. agreement with the pixel `ScreenSampler` on a smooth field,
  3. wind recovery from the prior-regularised profile likelihood, vs the pixel one.
"""

import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt
import numpy as np
from screen_operator import FourierScreenOperator, half_plane_index, screen_coords, von_karman

from furax import DiagonalOperator, HomothetyOperator
from furax.obs.atmosphere import ScreenSampler, profile_neg_log_likelihood
from furax.obs.coords import azel_to_quat, xieta_to_quat
from furax.obs.landscapes import TangentialLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import StokesI
from furax.tree import dot

jax.config.update('jax_enable_x64', True)


# --- geometry: constant-elevation az sweep, small focal plane --------------------------------
HEIGHT = 1000.0  # m
NDET, NSAMP, RATE = 4, 400, 4.0
times = jnp.arange(NSAMP) / RATE
az = jnp.deg2rad(10.0) * jnp.sin(2 * jnp.pi * times / 20.0)  # +-10 deg, 20 s period
qbore = azel_to_quat(az, jnp.deg2rad(50.0))
key = jax.random.key(0)
k_det, k_atm, k_noise = jax.random.split(key, 3)
xi, eta = jnp.deg2rad(2.0) * jax.random.uniform(k_det, (2, NDET), minval=-1, maxval=1)
qdet = xieta_to_quat(xi, eta)
u = screen_coords(qbore, qdet)
wind_true = jnp.array([3.0, -1.5])  # m/s
omega_true = wind_true / HEIGHT

# --- Fourier box: covers all visited y plus margin --------------------------------------------
# The series is periodic, so only the box size matters, not where it sits.
y_true = u + times[:, None] * omega_true
lo, hi = y_true.min((0, 1)), y_true.max((0, 1))
CENTRE = np.asarray((lo + hi) / 2)
BOX = float(jnp.max(hi - lo)) + 0.3  # dimensionless (units of height)
Q_MAX = 2 * np.pi / 0.1  # resolve ~100 m at 1 km
index = half_plane_index(BOX, Q_MAX)
modes = 2 * np.pi / BOX * index
S = von_karman(modes, amplitude=1.0, q0=2 * np.pi / 0.5)
S = S / S.max()
print(f'box={BOX:.2f}, modes M={modes.shape[0]} (x2 for cos/sin), samples={NDET * NSAMP}')

a_true = jnp.sqrt(S) * jax.random.normal(k_atm, (2, modes.shape[0]))


def fourier_op(omega):
    return FourierScreenOperator.create(index, BOX, qbore, qdet, times, omega)


P = fourier_op(omega_true)

# 1. transpose
d_rand = jax.random.normal(k_noise, P.out_structure.shape)
a_rand = jax.random.normal(k_atm, P.in_structure.shape)
lhs, rhs = dot(P(a_rand), d_rand), dot(a_rand, P.T(d_rand))
print(f'1. transpose: rel err {abs(lhs - rhs) / abs(lhs):.1e}')

# 2. vs pixel ScreenSampler on the same field, rasterised
signal = P(a_true)


def pixel_op(dx, wind):
    x0, y0 = (CENTRE * HEIGHT).tolist()
    landscape = TangentialLandscape.from_extent(
        BOX * HEIGHT, BOX * HEIGHT, dx, dx, HEIGHT, x0=x0, y0=y0
    )
    sampler = ScreenSampler.from_wind(qbore, qdet, wind, times, interpolate=True)
    return landscape, PointingOperator.from_sampler(landscape, sampler)


for dx in (40.0, 10.0):
    landscape, Ppix = pixel_op(dx, wind_true)
    ny, nx = landscape.shape
    px, py = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny))
    centres = jnp.stack([(px - nx / 2 + 0.5) * dx, (py - ny / 2 + 0.5) * dx], -1) / HEIGHT
    centres = centres + CENTRE
    phase = centres @ modes.T
    raster = jnp.cos(phase) @ a_true[0] + jnp.sin(phase) @ a_true[1]
    tod_pix = Ppix(StokesI(raster)).i
    err = jnp.std(tod_pix - signal) / jnp.std(signal)
    print(f'2. pixel dx={dx:4.0f} m vs Fourier: rel rms diff {err:.1e}')

# 3. wind recovery, white noise
sigma = 0.1 * float(jnp.std(signal))
d = signal + sigma * jax.random.normal(k_noise, signal.shape)
Ninv = HomothetyOperator(sigma**-2, in_structure=P.out_structure)
solver = lx.CG(rtol=1e-6, atol=1e-6, max_steps=200)


def fourier_nll(omega):
    """-b^T (P^T N^-1 P + S^-1)^-1 b: the profile chi^2 with the prior on."""
    P = fourier_op(omega)
    b = (P.T @ Ninv)(d)
    Sinv = DiagonalOperator(jnp.broadcast_to(1 / S, b.shape), in_structure=P.in_structure)
    A = P.T @ Ninv @ P + Sinv
    # shortcut: dense solve, fine for M ~ 100s; CG needs a coverage-aware preconditioner.
    return -b.ravel() @ jnp.linalg.solve(A.as_matrix(), b.ravel())


def pixel_nll(wind, dx):
    _, Ppix = pixel_op(dx, wind)
    Ninv_pix = HomothetyOperator(sigma**-2, in_structure=Ppix.out_structure)
    return profile_neg_log_likelihood(Ppix, StokesI(d), Ninv_pix, solver=solver)


fourier_nll = jax.jit(fourier_nll)
g = jax.grad(fourier_nll)(omega_true)
import time

t0 = time.time()
print(
    f'   fourier nll at truth {fourier_nll(omega_true):.4e} ({time.time() - t0:.1f}s)', flush=True
)
print(f'3. grad at truth (should be ~0 relative to scale): {g}')

vx = np.linspace(1.0, 5.0, 11)
curves = {'Fourier': [float(fourier_nll(jnp.array([v, wind_true[1]]) / HEIGHT)) for v in vx]}
for dx in (40.0,):
    f = jax.jit(lambda w, dx=dx: pixel_nll(w, dx))
    t0 = time.time()
    f(wind_true)
    print(f'   pixel dx={dx} one eval {time.time() - t0:.1f}s', flush=True)
    curves[f'pixel dx={dx:.0f} m'] = [float(f(jnp.array([v, wind_true[1]]))) for v in vx]

fig, ax = plt.subplots()
for name, c in curves.items():
    c = np.array(c)
    ax.plot(vx, (c - c.min()) / (c.max() - c.min()), label=name)
    print(f'   {name:>14}: argmin vx = {vx[np.argmin(c)]:.2f} (true {wind_true[0]})')
ax.axvline(wind_true[0], color='k', ls=':')
ax.set_xlabel('vx [m/s]')
ax.set_ylabel('normalised profile NLL')
ax.legend()
fig.savefig(__file__.replace('.py', '.png'))

# 4. conditioning of the amplitude solve vs coverage (dense, toy sizes only)
print('4. cond(P^T N^-1 P + S^-1) [Jacobi-preconditioned]:')
for ndet, nsamp in [(4, 400), (16, 400), (4, 1600), (16, 1600)]:
    t = jnp.arange(nsamp) / RATE
    qb = azel_to_quat(jnp.deg2rad(10.0) * jnp.sin(2 * jnp.pi * t / 20.0), jnp.deg2rad(50.0))
    xi, eta = jnp.deg2rad(2.0) * jax.random.uniform(k_det, (2, ndet), minval=-1, maxval=1)
    Pc = FourierScreenOperator.create(index, BOX, qb, xieta_to_quat(xi, eta), t, omega_true)
    phi = Pc.as_matrix()
    A = (phi.T @ phi) * sigma**-2 + jnp.diag(jnp.broadcast_to(1 / S, (2, S.size)).ravel())
    jac = 1 / jnp.sqrt(jnp.diag(A))
    ev = jnp.linalg.eigvalsh(A)
    evp = jnp.linalg.eigvalsh(jac[:, None] * A * jac[None])
    print(f'   ndet={ndet:2d} nsamp={nsamp:4d}: {ev[-1] / ev[0]:.1e} [{evp[-1] / evp[0]:.1e}]')
