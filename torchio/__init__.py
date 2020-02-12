"""Top-level package for torchio."""

__author__ = """Fernando Perez-Garcia"""
__email__ = 'fernando.perezgarcia.17@ucl.ac.uk'
__version__ = '0.12.13'

from . import utils
from .torchio import *
from .transforms import *
from .data import io, sampler, inference, ImagesDataset, Image, Queue, Subject
