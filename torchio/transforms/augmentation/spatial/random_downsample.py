from typing import Union, Tuple, Optional, List
import torch
from ....torchio import TypeRangeFloat
from ....data.subject import Subject
from ....utils import to_tuple
from .. import RandomTransform
from ...preprocessing import Resample


class RandomDownsample(RandomTransform):
    """Downsample an image along an axis.

    This transform simulates an image that has been acquired using anisotropic
    spacing, using downsampling with nearest neighbor interpolation.

    Args:
        axes: Axis or tuple of axes along which the image will be downsampled.
        downsampling: Downsampling factor :math:`m \gt 1`. If a tuple
            :math:`(a, b)` is provided then :math:`m \sim \mathcal{U}(a, b)`.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.

    Example:
        >>> from torchio import RandomDownsample
        >>> from torchio.datasets import Colin27
        >>> transform = RandomDownsample(axes=1, downsampling=2.)   # Multiply spacing of second axis by 2
        >>> transform = RandomDownsample(
        ...     axes=(0, 1, 2), downsampling=(2, 5)
        ... )   # Multiply spacing of one of the 3 axes by a factor randomly chosen in [2, 5]
        >>> colin = Colin27
        >>> transformed = transform(colin)  # images have now anisotropic spacing
    """

    def __init__(
            self,
            axes: Union[int, Tuple[int, ...]] = (0, 1, 2),
            downsampling: TypeRangeFloat = (1.5, 5),
            p: float = 1,
            seed: Optional[int] = None,
            ):
        super().__init__(p=p, seed=seed)
        self.axes = self.parse_axes(axes)
        self.downsampling_range = self.parse_range(
            downsampling, 'downsampling', min_constraint=1)

    @staticmethod
    def get_params(
            axes: Tuple[int, ...],
            downsampling_range: Tuple[float, float],
            ) -> List[bool]:
        axis = axes[torch.randint(0, len(axes), (1,))]
        downsampling = torch.FloatTensor(1).uniform_(*downsampling_range).item()
        return axis, downsampling

    @staticmethod
    def parse_axes(axes: Union[int, Tuple[int, ...]]):
        axes_tuple = to_tuple(axes)
        for axis in axes_tuple:
            is_int = isinstance(axis, int)
            if not is_int or axis not in (0, 1, 2):
                raise ValueError('All axes must be 0, 1 or 2')
        return axes_tuple

    def apply_transform(self, sample: Subject) -> Subject:
        axis, downsampling = self.get_params(self.axes, self.downsampling_range)
        random_parameters_dict = {'axis': axis, 'downsampling': downsampling}

        target_spacing = list(sample.spacing)
        target_spacing[axis] *= downsampling
        transform = Resample(
            tuple(target_spacing),
            image_interpolation='nearest',
            copy=False,  # already copied in super().__init__
        )
        sample = transform(sample)
        sample.add_transform(self, random_parameters_dict)
        return sample
