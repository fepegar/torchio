"""Main module."""

from pathlib import Path
from typing import Union, Tuple, Callable
import torch
import numpy as np

__all__ = [
    'INTENSITY',
    'LABEL',
    'SAMPLING_MAP',
    'DATA',
    'AFFINE',
    'IMAGE',
    'LOCATION',
    'TypePath',
    'TypeNumber',
    'TypeData',
    'TypeTuple',
    'TypeCallable',
]

INTENSITY = 'intensity'
LABEL = 'label'
SAMPLING_MAP = 'sampling_map'

DATA = 'data'
AFFINE = 'affine'

# For aggregator
IMAGE = 'image'
LOCATION = 'location'

TypePath = Union[Path, str]
TypeNumber = Union[int, float]
TypeData = Union[torch.Tensor, np.ndarray]
TypeTuple = Union[int, Tuple[int, int, int]]
TypeCallable = Callable[[torch.Tensor], torch.Tensor]
