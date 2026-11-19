import os

# With mpi4py installed, transitive sotodlib dependencies call MPI_Init. Fortunately we can opt out
# of this behaviour with `DISABLE_MPI` (read by pixell.mpi) and `MPI_DISABLE` (read by flacarray,
# imported via AxisManager.load).
os.environ.setdefault('DISABLE_MPI', '1')
os.environ.setdefault('MPI_DISABLE', '1')

# XLA:CPU runs the while loops it deems small on a single thread, and since jaxlib 0.12.1 that
# includes loops with a scatter, such as the batches of the pointing transpose, which then run
# several times slower (openxla/xla#53475). A zero threshold keeps every loop on the thread pool.
# Set before JAX creates its CPU client; extra backend options set by the user are left alone.
if 'xla_backend_extra_options' not in os.environ.get('XLA_FLAGS', ''):
    os.environ['XLA_FLAGS'] = ' '.join(
        filter(
            None,
            [
                os.environ.get('XLA_FLAGS'),
                '--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=0',
            ],
        )
    )
