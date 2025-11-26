import logging
import os
import sys

__all__ = [
    'logger',
]

logger = logging.getLogger('furax-mapmaking')
logger.setLevel(os.getenv('LOGLEVEL', 'INFO').upper())

# Add handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)  # Handler accepts all levels
    formatter = logging.Formatter('%(levelname)s [%(name)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False  # Don't propagate to root to avoid duplicate messages
