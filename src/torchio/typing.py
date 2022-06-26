import os
from typing import Union, Tuple, Callable, Optional, Sequence

import torch
import numpy as np


# For typing hints
TypePath = Union[str, os.PathLike]  # https://www.python.org/dev/peps/pep-0519/#provide-specific-type-hinting-support  # noqa: E501
TypeNumber = Union[int, float]
TypeKeys = Optional[Sequence[str]]
TypeData = Union[torch.Tensor, np.ndarray]
TypeDataAffine = Tuple[torch.Tensor, np.ndarray]

TypeDoubletInt = Tuple[int, int]
TypeTripletInt = Tuple[int, int, int]
TypeQuartetInt = Tuple[int, int, int, int]
TypeSextetInt = Tuple[int, int, int, int, int, int]

TypeTripletFloat = Tuple[float, float, float]
TypeSextetFloat = Tuple[float, float, float, float, float, float]

TypeTuple = Union[int, TypeTripletInt]
TypeRangeInt = Union[int, TypeDoubletInt]
TypeSpatialShape = Union[int, TypeTripletInt]
TypeRangeFloat = Union[float, Tuple[float, float]]
TypeCallable = Callable[[torch.Tensor], torch.Tensor]
TypeDirection2D = Tuple[float, float, float, float]
TypeDirection3D = Tuple[
    float, float, float, float, float, float, float, float, float,
]
TypeDirection = Union[TypeDirection2D, TypeDirection3D]
