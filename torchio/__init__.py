"""Top-level package for torchio."""

__author__ = """Fernando Perez-Garcia"""
__email__ = 'fernando.perezgarcia.17@ucl.ac.uk'
__version__ = '0.13.19'

from . import utils
from .torchio import *
from .transforms import *
from .data import io, sampler, inference, ImagesDataset, Image, Queue, Subject
from . import datasets

print('If you use TorchIO for your research, please cite the following paper:')
print('\nPérez-García et al., TorchIO: a Python library for efficient loading,')
print('preprocessing, augmentation and patch-based sampling of medical images')
print('in deep learning')
print('(https://arxiv.org/abs/2003.04696)')
