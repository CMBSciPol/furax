import os

# With mpi4py installed, transitive sotodlib dependencies call MPI_Init. Fortunately we can opt out
# of this behaviour with `DISABLE_MPI` (read by pixell.mpi) and `MPI_DISABLE` (read by flacarray,
# imported via AxisManager.load).
os.environ.setdefault('DISABLE_MPI', '1')
os.environ.setdefault('MPI_DISABLE', '1')
