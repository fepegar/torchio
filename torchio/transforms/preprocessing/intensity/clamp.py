from torchio.data.image import ScalarImage
from typing import Optional
import torch
from ....data.subject import Subject
from ...intensity_transform import IntensityTransform


class Clamp(IntensityTransform):
    """Clamp intensity values to a certain range.
    Args:
        out_min: See :func:`torch.clamp`.
        out_max: See :func:`torch.clamp`.

    Example:
        >>> import torchio as tio
        >>> ct = tio.ScalarImage('ct_scan.nii.gz')
        >>> ct_air, ct_bone = -1000, 1000
        >>> clamp = tio.Clamp(out_min=ct_air, out_max=ct_bone)
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

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            self.apply_clamp(image)
        return subject

    def apply_clamp(self, image: ScalarImage):
        image.set_data(self.clamp(image.data))

    def clamp(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clamp(self.out_min, self.out_max)
