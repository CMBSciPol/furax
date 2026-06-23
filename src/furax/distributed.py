import jax
import jax.experimental.multihost_utils as mhu


def maybe_init() -> bool:
    try:
        jax.distributed.initialize()
    except ValueError:
        return False
    mhu.sync_global_devices('furax.startup')
    return True
