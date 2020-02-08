from typing import Union, Tuple, Callable
import SimpleITK as sitk
from .bounds_transform import BoundsTransform


class Crop(BoundsTransform):
    def __init__(
            self,
            cropping: Union[int, Tuple[int, int, int]],
            verbose: bool = False,
            ):
        super().__init__(cropping, verbose=verbose)

    @property
    def bounds_function(self) -> Callable:
        return sitk.Crop
