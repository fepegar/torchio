from typing import Callable
import SimpleITK as sitk
from .bounds_transform import BoundsTransform


class Crop(BoundsTransform):
    r"""Crop an image.

    Args:
        cropping: Tuple
            :math:`(d_{ini}, d_{fin}, h_{ini}, h_{fin}, w_{ini}, w_{fin})`
            defining the number of values cropped from the edges of each axis.
            If the initial shape of the image is
            :math:`D \times H \times W`, the final shape will be
            :math:`(- d_{ini} + D - d_{fin}) \times (- h_{ini} + H - h_{fin}) \times (- w_{ini} + W - w_{fin})`.
            If only three values :math:`(d, h, w)` are provided, then
            :math:`d_{ini} = d_{fin} = d`,
            :math:`h_{ini} = h_{fin} = h` and
            :math:`w_{ini} = w_{fin} = w`.
            If only one value :math:`n` is provided, then
            :math:`d_{ini} = d_{fin} = h_{ini} = h_{fin} = w_{ini} = w_{fin} = n`.

    """
    @property
    def bounds_function(self) -> Callable:
        return sitk.Crop
