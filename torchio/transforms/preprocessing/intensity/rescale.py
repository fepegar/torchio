import warnings
from typing import Optional, List

import torch
import numpy as np
from deprecated import deprecated

from ....data.subject import Subject
from ....torchio import DATA, TypeRangeFloat
from .normalization_transform import NormalizationTransform, TypeMaskingMethod


class RescaleIntensity(NormalizationTransform):
    """Rescale intensity values to a certain range.

    Args:
        out_min_max: Range :math:`(n_{min}, n_{max})` of output intensities.
            If only one value :math:`d` is provided,
            :math:`(n_{min}, n_{max}) = (-d, d)`.
        percentiles: Percentile values of the input image that will be mapped
            to :math:`(n_{min}, n_{max})`. They can be used for contrast
            stretching, as in `this scikit-image example`_. For example,
            Isensee et al. use ``(0.05, 99.5)`` in their `nn-UNet paper`_.
            If only one value :math:`d` is provided,
            :math:`(n_{min}, n_{max}) = (0, d)`.
        masking_method: See
            :py:class:`~torchio.transforms.preprocessing.normalization_transform.NormalizationTransform`.
        p: Probability that this transform will be applied.
        keys: See :py:class:`~torchio.transforms.Transform`.

    .. _this scikit-image example: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
    .. _nn-UNet paper: https://arxiv.org/abs/1809.10486
    """
    def __init__(
            self,
            out_min_max: TypeRangeFloat,
            percentiles: TypeRangeFloat = (0, 100),
            masking_method: TypeMaskingMethod = None,
            p: float = 1,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(masking_method=masking_method, p=p, keys=keys)
        self.out_min, self.out_max = self.parse_range(
            out_min_max, 'out_min_max')
        self.percentiles = self.parse_range(
            percentiles, 'percentiles', min_constraint=0, max_constraint=100)

    def apply_normalization(
            self,
            sample: Subject,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        image_dict = sample[image_name]
        image_dict[DATA] = self.rescale(image_dict[DATA], mask, image_name)

    def rescale(
            self,
            tensor: torch.Tensor,
            mask: torch.Tensor,
            image_name: str,
            ) -> torch.Tensor:
        array = tensor.clone().numpy()
        mask = mask.numpy()
        values = array[mask]
        cutoff = np.percentile(values, self.percentiles)
        np.clip(array, *cutoff, out=array)
        array -= array.min()  # [0, max]
        array_max = array.max()
        if array_max == 0:
            message = (
                f'Rescaling image "{image_name}" not possible'
                ' due to division by zero'
            )
            warnings.warn(message)
            return tensor
        array /= array.max()  # [0, 1]
        out_range = self.out_max - self.out_min
        array *= out_range  # [0, out_range]
        array += self.out_min  # [out_min, out_max]
        return torch.from_numpy(array)


@deprecated('Rescale is deprecated. Use RescaleIntensity instead')
class Rescale(RescaleIntensity):
    pass
