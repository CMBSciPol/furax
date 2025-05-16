import logging
import os

__all__ = [
    'logger',
]

logger = logging.getLogger('furax-mapmaking')
logger.setLevel(os.getenv('LOGLEVEL', 'INFO').upper())
