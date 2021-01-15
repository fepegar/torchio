import warnings
from typing import Union, Tuple, List

import torch

from ....typing import TypeRangeFloat
from ....data.subject import Subject
from ....utils import to_tuple
from .. import RandomTransform
from ...preprocessing import Resample


class RandomAnisotropy(RandomTransform):
    r"""Downsample an image along an axis and upsample to initial space.

    This transform simulates an image that has been acquired using anisotropic
    spacing and resampled back to its original spacing.

    Similar to the work by Billot et al.: `Partial Volume Segmentation of Brain
    MRI Scans of any Resolution and
    Contrast <billot>`_.

    Args:
        axes: Axis or tuple of axes along which the image will be downsampled.
        downsampling: Downsampling factor :math:`m \gt 1`. If a tuple
            :math:`(a, b)` is provided then :math:`m \sim \mathcal{U}(a, b)`.
        image_interpolation: Image interpolation used to upsample the image
            back to its initial spacing. Downsampling is performed using
            nearest neighbor interpolation. See :ref:`Interpolation` for
            supported interpolation types.
        scalars_only: Apply only to instances of :class:`torchio.ScalarImage`.
            This is useful when the segmentation quality needs to be kept,
            as in `Billot et al. <billot>`_.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. _billot: https://link.springer.com/chapter/10.1007/978-3-030-59728-3_18

    Example:
        >>> import torchio as tio
        >>> transform = tio.RandomAnisotropy(axes=1, downsampling=2)
        >>> transform = tio.RandomAnisotropy(
        ...     axes=(0, 1, 2),
        ...     downsampling=(2, 5),
        ... )   # Multiply spacing of one of the 3 axes by a factor randomly chosen in [2, 5]
        >>> colin = tio.datasets.Colin27()
        >>> transformed = transform(colin)
    """  # noqa: E501

    def __init__(
            self,
            axes: Union[int, Tuple[int, ...]] = (0, 1, 2),
            downsampling: TypeRangeFloat = (1.5, 5),
            image_interpolation: str = 'linear',
            scalars_only: bool = True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.axes = self.parse_axes(axes)
        self.downsampling_range = self._parse_range(
            downsampling, 'downsampling', min_constraint=1)
        parsed_interpolation = self.parse_interpolation(image_interpolation)
        self.image_interpolation = parsed_interpolation
        self.scalars_only = scalars_only

    def get_params(
            self,
            axes: Tuple[int, ...],
            downsampling_range: Tuple[float, float],
            is_2d: bool,
            ) -> List[bool]:
        axis = axes[torch.randint(0, len(axes), (1,))]
        downsampling = self.sample_uniform(*downsampling_range).item()
        return axis, downsampling

    @staticmethod
    def parse_axes(axes: Union[int, Tuple[int, ...]]):
        axes_tuple = to_tuple(axes)
        for axis in axes_tuple:
            is_int = isinstance(axis, int)
            if not is_int or axis not in (0, 1, 2):
                raise ValueError('All axes must be 0, 1 or 2')
        return axes_tuple

    def apply_transform(self, subject: Subject) -> Subject:
        is_2d = subject.get_first_image().is_2d()
        if is_2d and 2 in self.axes:
            warnings.warn(
                f'Input image is 2D, but "2" is in axes: {self.axes}',
                RuntimeWarning,
            )
            self.axes = list(self.axes)
            self.axes.remove(2)
        axis, downsampling = self.get_params(
            self.axes,
            self.downsampling_range,
            is_2d,
        )
        target_spacing = list(subject.spacing)
        target_spacing[axis] *= downsampling

        arguments = {
            'image_interpolation': 'nearest',
            'scalars_only': self.scalars_only,
        }

        downsample = Resample(
            tuple(target_spacing),
            **self.add_include_exclude(arguments)
        )
        downsampled = downsample(subject)
        upsample = Resample(
            subject.get_first_image(),
            image_interpolation=self.image_interpolation,
            scalars_only=self.scalars_only,
        )
        upsampled = upsample(downsampled)
        return upsampled
