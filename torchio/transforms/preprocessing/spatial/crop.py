from typing import Union, Tuple, Callable
import SimpleITK as sitk
from .bounds_transform import BoundsTransform


class Crop(BoundsTransform):
    """Crop an image.

    Args:
        cropping: Tuple
            :math:`(D_{ini}, D_{fin}, H_{ini}, H_{fin}, W_{ini}, W_{fin})`
            defining the number of values cropped from the edges of each axis.
            If only three values :math:`(D, H, W)` are provided, then
            :math:`D_{ini} = D_{fin} = D`,
            :math:`H_{ini} = H_{fin} = H` and
            :math:`W_{ini} = W_{fin} = W`.
            If only one value :math:`N` is provided, then
            :math:`D_{ini} = D_{fin} = H_{ini} = H_{fin} = W_{ini} = W_{fin} = N`.
        verbose:

    """
    def __init__(
            self,
            cropping: Union[int, Tuple[int, int, int]],
            verbose: bool = False,
            ):
        super().__init__(cropping, verbose=verbose)

    @property
    def bounds_function(self) -> Callable:
        return sitk.Crop
