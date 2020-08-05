"""Top-level package for torchio."""

__author__ = """Fernando Perez-Garcia"""
__email__ = 'fernando.perezgarcia.17@ucl.ac.uk'
__version__ = '0.17.21'

import os
from . import utils
from .torchio import *
from .transforms import *
from .data import (
    io,
    sampler,
    inference,
    ImagesDataset,
    Image,
    ScalarImage,
    LabelMap,
    Queue,
    Subject,
)
from . import datasets
from . import reference

CITATION = """If you use TorchIO for your research, please cite the following paper:
Pérez-García et al., TorchIO: a Python library for efficient loading,
preprocessing, augmentation and patch-based sampling of medical images
in deep learning. Link: https://arxiv.org/abs/2003.04696
"""

# Thanks for citing torchio. Without citations, researchers will not use TorchIO
if 'TORCHIO_HIDE_CITATION_PROMPT' not in os.environ:
    print(CITATION)
