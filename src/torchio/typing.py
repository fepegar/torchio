from collections.abc import Sequence
from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
import torch

# For typing hints
TypePath = Union[str, Path]
TypeNumber = Union[int, float]
TypeKeys = Optional[Sequence[str]]
TypeData = Union[torch.Tensor, np.ndarray]
TypeDataAffine = tuple[torch.Tensor, np.ndarray]
TypeSlice = Union[int, slice]

TypeDoubletInt = tuple[int, int]
TypeTripletInt = tuple[int, int, int]
TypeQuartetInt = tuple[int, int, int, int]
TypeSextetInt = tuple[int, int, int, int, int, int]

TypeDoubleFloat = tuple[float, float]
TypeTripletFloat = tuple[float, float, float]
TypeSextetFloat = tuple[float, float, float, float, float, float]

TypeTuple = Union[int, TypeTripletInt]
TypeRangeInt = Union[int, TypeDoubletInt]
TypeSpatialShape = Union[int, TypeTripletInt]
TypeRangeFloat = Union[float, tuple[float, float]]
TypeCallable = Callable[[torch.Tensor], torch.Tensor]
TypeDirection2D = tuple[float, float, float, float]
TypeDirection3D = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]
TypeDirection = Union[TypeDirection2D, TypeDirection3D]
