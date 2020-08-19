from typing import Callable
import SimpleITK as sitk
from .bounds_transform import BoundsTransform


class Crop(BoundsTransform):
    r"""Crop an image.

    Args:
        cropping: Tuple
            :math:`(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})`
            defining the number of values cropped from the edges of each axis.
            If the initial shape of the image is
            :math:`H \times W \times D`, the final shape will be
            :math:`(- w_{ini} + W - w_{fin}) \times (- h_{ini} + H - h_{fin})
            \times (- d_{ini} + D - d_{fin})`.
            If only three values :math:`(w, h, d)` are provided, then
            :math:`w_{ini} = w_{fin} = w`,
            :math:`h_{ini} = h_{fin} = h` and
            :math:`d_{ini} = d_{fin} = d`.
            If only one value :math:`n` is provided, then
            :math:`w_{ini} = w_{fin} = h_{ini} = h_{fin}
            = d_{ini} = d_{fin} = n`.

    """
    @property
    def bounds_function(self) -> Callable:
        return sitk.Crop
