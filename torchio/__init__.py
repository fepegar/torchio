"""Top-level package for torchio."""

__author__ = """Fernando Perez-Garcia"""
__email__ = 'fernando.perezgarcia.17@ucl.ac.uk'
__version__ = '0.18.7'

from . import utils
from .torchio import *  # noqa: F401, F403
from .transforms import *  # noqa: F401, F403
from .data import (
    io,
    sampler,
    inference,
    SubjectsDataset,
    Image,
    ScalarImage,
    LabelMap,
    Queue,
    Subject,
)
from . import datasets
from . import reference


__all__ = [
    'utils',
    'io',
    'sampler',
    'inference',
    'SubjectsDataset',
    'Image',
    'ScalarImage',
    'LabelMap',
    'Queue',
    'Subject',
    'datasets',
    'reference',
]
