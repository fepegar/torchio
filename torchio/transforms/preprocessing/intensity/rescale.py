import warnings

import torch
import numpy as np

from ....data.subject import Subject
from ....typing import TypeRangeFloat
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
            Isensee et al. use ``(0.5, 99.5)`` in their `nn-UNet paper`_.
            If only one value :math:`d` is provided,
            :math:`(n_{min}, n_{max}) = (0, d)`.
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. _this scikit-image example: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
    .. _nn-UNet paper: https://arxiv.org/abs/1809.10486
    """  # noqa: E501
    def __init__(
            self,
            out_min_max: TypeRangeFloat = (0, 1),
            percentiles: TypeRangeFloat = (0, 100),
            masking_method: TypeMaskingMethod = None,
            **kwargs
            ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.out_min_max = out_min_max
        self.out_min, self.out_max = self._parse_range(
            out_min_max, 'out_min_max')
        self.percentiles = self._parse_range(
            percentiles, 'percentiles', min_constraint=0, max_constraint=100)
        self.args_names = 'out_min_max', 'percentiles', 'masking_method'

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        image = subject[image_name]
        image.set_data(self.rescale(image.data, mask, image_name))

    def rescale(
            self,
            tensor: torch.Tensor,
            mask: torch.Tensor,
            image_name: str,
            ) -> torch.Tensor:
        # The tensor is cloned as in-place operations will be used
        array = tensor.clone().float().numpy()
        mask = mask.numpy()
        values = array[mask]
        cutoff = np.percentile(values, self.percentiles)
        np.clip(array, *cutoff, out=array)
        array -= array.min()  # [0, max]
        array_max = array.max()  # waiting for walrus operator
        if array_max == 0:  # should this be compared using a tolerance?
            message = (
                f'Rescaling image "{image_name}" not possible'
                ' due to division by zero'
            )
            warnings.warn(message, RuntimeWarning)
            return tensor
        array /= array_max  # [0, 1]
        out_range = self.out_max - self.out_min
        array *= out_range  # [0, out_range]
        array += self.out_min  # [out_min, out_max]
        return torch.from_numpy(array)
