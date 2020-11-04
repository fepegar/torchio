import warnings
from typing import Optional, List

import torch
import numpy as np
from deprecated import deprecated

from ....data.subject import Subject
from ....torchio import DATA, TypeRangeFloat
from .normalization_transform import NormalizationTransform, TypeMaskingMethod


class ThresholdIntensity(NormalizationTransform):
    """Threshold intensity values between a certain range.

    Args:
        out_min_max: Range :math:`(n_{min}, n_{max})` of output intensities.
            If only one value :math:`d` is provided,
            :math:`(n_{min}, n_{max}) = (-d, d)`.
        masking_method: See
            :py:class:`~torchio.transforms.preprocessing.normalization_transform.NormalizationTransform`.
        p: Probability that this transform will be applied.
        keys: See :py:class:`~torchio.transforms.Transform`.

    """
    def __init__(
            self,
            out_min_max: TypeRangeFloat,
            masking_method: TypeMaskingMethod = None,
            p: float = 1,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(masking_method=masking_method, p=p, keys=keys)
        self.out_min, self.out_max = self.parse_range(
            out_min_max, 'out_min_max')

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        image_dict = subject[image_name]
        image_dict[DATA] = self.clip(image_dict[DATA], mask, image_name)

    def clip(
            self,
            tensor: torch.Tensor,
            mask: torch.Tensor,
            image_name: str,
            ) -> torch.Tensor:
        array = tensor.clone().numpy()
        mask = mask.numpy()
        values = array[mask]
        values = [0 if a_ > self.out_max else a_ for a_ in values]
        values = [0 if a_ < self.out_min else a_ for a_ in values]
        return torch.from_numpy(values)
