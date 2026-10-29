import threading

import jax.numpy as jnp
import numpy as np

from furax.mapmaking._checkpoint import SolveCheckpoint, remove_checkpoint_files


def test_concurrent_saves_keep_the_latest_iterate(tmp_path):
    checkpoint = SolveCheckpoint(tmp_path / 'checkpoint.npz')
    threads = [
        threading.Thread(target=checkpoint.save, args=(n, jnp.full(3, float(n))))
        for n in (8, 4, 6, 2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    like = jnp.zeros(3)
    x, n_steps = SolveCheckpoint(checkpoint.path).load(like)
    assert n_steps == 8
    np.testing.assert_array_equal(x, 8.0)


def test_remove_deletes_partial_writes(tmp_path):
    path = tmp_path / 'checkpoint.npz'
    checkpoint = SolveCheckpoint(path)
    checkpoint.save(1, jnp.ones(2))
    partial = tmp_path / 'checkpoint.npz.tmp.123'
    partial.touch()
    other = tmp_path / 'map.fits'
    other.touch()
    checkpoint.remove()
    assert not path.exists()
    assert not partial.exists()
    assert other.exists()


def test_remove_checkpoint_files_covers_both_solves(tmp_path):
    names = ['checkpoint.npz', 'checkpoint_injected.npz', 'checkpoint_injected.npz.tmp.7']
    for name in names:
        (tmp_path / name).touch()
    remove_checkpoint_files(tmp_path)
    assert list(tmp_path.iterdir()) == []
