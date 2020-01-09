"""Top-level package for torchio."""

__author__ = """Fernando Perez-Garcia"""
__email__ = 'fernando.perezgarcia.17@ucl.ac.uk'
__version__ = '0.8.1'

from . import io
from . import utils
from . import sampler
from . import inference
from .torchio import *
from .queue import Queue
from .transforms import *
from .dataset import ImagesDataset, Image
