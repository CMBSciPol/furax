# Intentionally empty: no eager re-exports.
#
# Re-exporting SOTODLibObservation / LazySOTODLibObservation here would force
# `observation.py` (and its jax-using transitive imports) to load at package
# import time, which initializes the JAX backend before
# `jax.distributed.initialize()` can run in distributed entrypoints.
# Import the concrete classes from `furax.interfaces.sotodlib.observation`
# directly.
