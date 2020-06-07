"""Main module."""

from pathlib import Path
from typing import Union, Tuple, Callable
import torch
import numpy as np


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

# In PyTorch convention
CHANNELS_DIMENSION = 1

# For typing hints
TypePath = Union[Path, str]
TypeNumber = Union[int, float]
TypeData = Union[torch.Tensor, np.ndarray]
TypeTripletInt = Tuple[int, int, int]
TypeTripletFloat = Tuple[float, float, float]
TypeTuple = Union[int, TypeTripletInt]
TypeRangeInt = Union[int, Tuple[int, int]]
TypePatchSize = Union[int, Tuple[int, int, int]]
TypeRangeFloat = Union[float, Tuple[float, float]]
TypeCallable = Callable[[torch.Tensor], torch.Tensor]
