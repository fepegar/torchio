from typing import Optional
import torch
from ....data.subject import Subject
from .normalization_transform import NormalizationTransform


class ClampIntensity(NormalizationTransform):
    """Clamp intensity values to a certain range.
    Args:
        out_min: See :func:`~torch.clamp`.
        out_max: See :func:`~torch.clamp`.

    Example:
        >>> import torchio as tio
        >>> ct = tio.ScalarImage('ct_scan.nii.gz')
        >>> ct_air, ct_bone = -1000, 1000
        >>> clamp = tio.ClampIntensity(out_min=ct_air, out_max=ct_bone)
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

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        image = subject[image_name]
        image.set_data(self.clamp(image.data, image_name))

    def clamp(
            self,
            tensor: torch.Tensor,
            image_name: str,
            ) -> torch.Tensor:
        return tensor.clamp(self.out_min, self.out_max)
