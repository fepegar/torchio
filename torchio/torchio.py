"""Main module."""

from pathlib import Path
from typing import Union, Tuple, Callable
import torch
import numpy as np

__all__ = [
    'INTENSITY',
    'LABEL',
    'SAMPLING_MAP',
    'PATH',
    'TYPE',
    'STEM',
    'DATA',
    'AFFINE',
    'IMAGE',
    'LOCATION',
    'TypePath',
    'TypeNumber',
    'TypeData',
    'TypeTuple',
    'TypeCallable',
    'TypeRangeInt',
    'TypeRangeFloat',
]

# Image types
INTENSITY = 'intensity'
LABEL = 'label'
SAMPLING_MAP = 'sampling_map'

# Keys for dataset samples
PATH = 'path'
TYPE = 'type'
STEM = 'stem'
DATA = 'data'
AFFINE = 'affine'

# For aggregator
IMAGE = 'image'
LOCATION = 'location'

# For typing hints
TypePath = Union[Path, str]
TypeNumber = Union[int, float]
TypeData = Union[torch.Tensor, np.ndarray]
TypeTuple = Union[int, Tuple[int, int, int]]
TypeRangeInt = Union[int, Tuple[int, int]]
TypeRangeFloat = Union[float, Tuple[float, float]]
TypeCallable = Callable[[torch.Tensor], torch.Tensor]
