from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch


# For typing hints
TypePath = Union[str, Path]
TypeNumber = Union[int, float]
TypeKeys = Optional[Sequence[str]]
TypeData = Union[torch.Tensor, np.ndarray]
TypeDataAffine = Tuple[torch.Tensor, np.ndarray]

TypeDoubletInt = Tuple[int, int]
TypeTripletInt = Tuple[int, int, int]
TypeQuartetInt = Tuple[int, int, int, int]
TypeSextetInt = Tuple[int, int, int, int, int, int]

TypeDoubleFloat = Tuple[float, float]
TypeTripletFloat = Tuple[float, float, float]
TypeSextetFloat = Tuple[float, float, float, float, float, float]

TypeTuple = Union[int, TypeTripletInt]
TypeRangeInt = Union[int, TypeDoubletInt]
TypeSpatialShape = Union[int, TypeTripletInt]
TypeRangeFloat = Union[float, Tuple[float, float]]
TypeCallable = Callable[[torch.Tensor], torch.Tensor]
TypeDirection2D = Tuple[float, float, float, float]
TypeDirection3D = Tuple[
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
