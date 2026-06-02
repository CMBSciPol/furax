import logging
import os

import jax

__all__ = [
    'logger',
]


class RankFilter(logging.Filter):
    """Inject the JAX process index as ``rank`` on every record.

    Resolved lazily at log time: before ``jax.distributed.initialize`` it is 0,
    afterwards the true per-process rank. This keeps the logger usable in both
    single- and multi-process runs without coupling logger import to JAX init.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = jax.process_index()
        return True


logger = logging.getLogger('furax-mapmaking')
logger.setLevel(os.getenv('LOGLEVEL', 'INFO').upper())

# Add handler if none exists
if not logger.handlers:
    logger.propagate = False
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [rank %(rank)d] - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addFilter(RankFilter())
    logger.addHandler(ch)
