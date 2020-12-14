from pathlib import Path
from typing import Union, Tuple, Callable, Optional, Sequence

import torch
import numpy as np


# For typing hints
TypePath = Union[Path, str]
TypeNumber = Union[int, float]
TypeKeys = Optional[Sequence[str]]
TypeData = Union[torch.Tensor, np.ndarray]
TypeTripletInt = Tuple[int, int, int]
TypeSextetInt = Tuple[int, int, int, int, int, int]
TypeTripletFloat = Tuple[float, float, float]
TypeSextetFloat = Tuple[float, float, float, float, float, float]
TypeTuple = Union[int, TypeTripletInt]
TypeRangeInt = Union[int, Tuple[int, int]]
TypePatchSize = Union[int, Tuple[int, int, int]]
TypeRangeFloat = Union[float, Tuple[float, float]]
TypeCallable = Callable[[torch.Tensor], torch.Tensor]
