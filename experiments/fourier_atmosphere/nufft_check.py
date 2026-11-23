"""FourierScreenOperator: NUFFT backend against the dense sum, then timings at growing sizes."""

import time

import jax
import jax.numpy as jnp
import numpy as np
from screen_operator import FourierScreenOperator, half_plane_index, screen_coords

from furax.obs.coords import azel_to_quat, xieta_to_quat
from furax.tree import dot

jax.config.update('jax_enable_x64', True)


def geometry(ndet, nsamp, rate=10.0, seed=0):
    times = jnp.arange(nsamp) / rate
    qbore = azel_to_quat(jnp.deg2rad(10.0) * jnp.sin(2 * jnp.pi * times / 20.0), jnp.deg2rad(50.0))
    xi, eta = jnp.deg2rad(2.0) * jax.random.uniform(jax.random.key(seed), (2, ndet), minval=-1)
    qdet = xieta_to_quat(xi, eta)
    u = screen_coords(qbore, qdet)
    box = float(jnp.max(u.max((0, 1)) - u.min((0, 1)))) + 0.01 * float(times[-1]) + 0.2
    return qbore, qdet, times, box


def ops(ndet, nsamp, resolution):
    qbore, qdet, times, box = geometry(ndet, nsamp)
    index = half_plane_index(box, 2 * np.pi / resolution)
    omega = jnp.array([0.004, -0.002])
    make = lambda backend, omega=omega: FourierScreenOperator.create(
        index, box, qbore, qdet, times, omega, backend
    )
    return make


# --- correctness ------------------------------------------------------------------------------
make = ops(8, 500, 0.1)
dense, fast = make('dense'), make('nufft')
key1, key2 = jax.random.split(jax.random.key(1))
a = jax.random.normal(key1, dense.in_structure.shape)
d = jax.random.normal(key2, dense.out_structure.shape)
print(f'M={dense.in_structure.shape[1]} modes, {d.size} samples')
print(
    f'forward  |nufft - dense| / |dense| = {jnp.linalg.norm(fast(a) - dense(a)) / jnp.linalg.norm(dense(a)):.1e}'
)
print(
    f'transpose |nufft - dense| / |dense| = {jnp.linalg.norm(fast.T(d) - dense.T(d)) / jnp.linalg.norm(dense.T(d)):.1e}'
)
print(
    f'adjoint test (nufft): {abs(dot(fast(a), d) - dot(a, fast.T(d))) / abs(dot(a, fast.T(d))):.1e}'
)


def loss(omega, backend):
    return jnp.sum((make(backend, omega)(a) - d) ** 2)


g_dense = jax.grad(loss)(dense.omega, 'dense')
g_fast = jax.grad(loss)(dense.omega, 'nufft')
print(
    f'd/d omega |nufft - dense| / |dense| = {jnp.linalg.norm(g_fast - g_dense) / jnp.linalg.norm(g_dense):.1e}'
)


# --- timings: P and P^T, jitted, after compilation --------------------------------------------
def bench(f, x, repeat=3):
    jax.block_until_ready(f(x))
    t0 = time.perf_counter()
    for _ in range(repeat):
        jax.block_until_ready(f(x))
    return (time.perf_counter() - t0) / repeat * 1e3


print(
    f'\n{"samples":>8} {"M":>6} | {"dense P":>9} {"nufft P":>9} | {"dense PT":>9} {"nufft PT":>9}  [ms]'
)
for ndet, nsamp, res in [
    (8, 500, 0.1),
    (16, 2000, 0.1),
    (16, 2000, 0.05),
    (32, 4000, 0.05),
    (32, 4000, 0.025),
    (64, 20000, 0.025),
]:
    make = ops(ndet, nsamp, res)
    row = []
    for backend in ('dense', 'nufft'):
        op = make(backend)
        if backend == 'dense' and ndet * nsamp * op.in_structure.shape[1] > 1e8:
            row.append((np.nan, np.nan))  # too slow / too much memory
            continue
        a = jnp.ones(op.in_structure.shape)
        d = jnp.ones(op.out_structure.shape)
        row.append(
            (bench(jax.jit(lambda x, op=op: op(x)), a), bench(jax.jit(lambda x, op=op: op.T(x)), d))
        )
    m = op.in_structure.shape[1]
    print(
        f'{ndet * nsamp:>8} {m:>6} | {row[0][0]:9.1f} {row[1][0]:9.1f} | {row[0][1]:9.1f} {row[1][1]:9.1f}',
        flush=True,
    )
