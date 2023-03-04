import warnings
from typing import Optional

import numpy as np
import torch

from ....data.subject import Subject
from ....typing import TypeRangeFloat
from .normalization_transform import NormalizationTransform
from .normalization_transform import TypeMaskingMethod


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
        in_min_max: Range :math:`(m_{min}, m_{max})` of input intensities that
            will be mapped to :math:`(n_{min}, n_{max})`. If ``None``, the
            minimum and maximum input intensities will be used.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> ct = tio.ScalarImage('ct_scan.nii.gz')
        >>> ct_air, ct_bone = -1000, 1000
        >>> rescale = tio.RescaleIntensity(
        ...     out_min_max=(-1, 1), in_min_max=(ct_air, ct_bone))
        >>> ct_normalized = rescale(ct)

    .. _this scikit-image example: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
    .. _nn-UNet paper: https://arxiv.org/abs/1809.10486
    """  # noqa: B950

    def __init__(
        self,
        out_min_max: TypeRangeFloat = (0, 1),
        percentiles: TypeRangeFloat = (0, 100),
        masking_method: TypeMaskingMethod = None,
        in_min_max: Optional[TypeRangeFloat] = None,
        **kwargs,
    ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.out_min_max = out_min_max
        self.in_min_max = in_min_max
        self.out_min, self.out_max = self._parse_range(
            out_min_max,
            'out_min_max',
        )
        self.percentiles = self._parse_range(
            percentiles,
            'percentiles',
            min_constraint=0,
            max_constraint=100,
        )

        self.in_min: Optional[float]
        self.in_max: Optional[float]
        if self.in_min_max is not None:
            self.in_min, self.in_max = self._parse_range(
                self.in_min_max,
                'in_min_max',
            )
        else:
            self.in_min = None
            self.in_max = None

        self.args_names = [
            'out_min_max',
            'percentiles',
            'masking_method',
            'in_min_max',
        ]
        self.invert_transform = False

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
        if not mask.any():
            message = (
                f'Rescaling image "{image_name}" not possible'
                ' because the mask to compute the statistics is empty'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return tensor
        values = array[mask]
        cutoff = np.percentile(values, self.percentiles)
        np.clip(array, *cutoff, out=array)  # type: ignore[call-overload]
        if self.in_min_max is None:
            self.in_min_max = self._parse_range(
                (array.min(), array.max()),
                'in_min_max',
            )
            self.in_min, self.in_max = self.in_min_max
        assert self.in_min is not None
        assert self.in_max is not None
        in_range = self.in_max - self.in_min
        if in_range == 0:  # should this be compared using a tolerance?
            message = (
                f'Rescaling image "{image_name}" not possible'
                ' because all the intensity values are the same'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return tensor
        out_range = self.out_max - self.out_min
        if self.invert_transform:
            array -= self.out_min
            array /= out_range
            array *= in_range
            array += self.in_min
        else:
            array -= self.in_min
            array /= in_range
            array *= out_range
            array += self.out_min
        return torch.as_tensor(array)
