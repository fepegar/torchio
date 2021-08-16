from torchio.data.image import ScalarImage
from typing import Optional
import torch
from ....data.subject import Subject
from ...intensity_transform import IntensityTransform


class Clamp(IntensityTransform):
    """Clamp intensity values into a range :math:`[a, b]`.

    For more information, see :func:`torch.clamp`.

    Args:
        out_min: Minimum value :math:`a` of the output image. If ``None``, the
            minimum of the image is used.
        out_max: Maximum value :math:`b` of the output image. If ``None``, the
            maximum of the image is used.

    Example:
        >>> import torchio as tio
        >>> ct = tio.ScalarImage('ct_scan.nii.gz')
        >>> HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 1000
        >>> clamp = tio.Clamp(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE)
        >>> ct_clamped = clamp(ct)
    """
    def __init__(
            self,
            out_min: Optional[float] = None,
            out_max: Optional[float] = None,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.out_min, self.out_max = out_min, out_max
        self.args_names = 'out_min', 'out_max'

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            self.apply_clamp(image)
        return subject

    def apply_clamp(self, image: ScalarImage) -> None:
        image.set_data(self.clamp(image.data))

    def clamp(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clamp(self.out_min, self.out_max)
