from typing import Union, Tuple, Optional, List
import torch
from ....torchio import DATA
from ....data.subject import Subject
from ....utils import to_tuple
from .. import RandomTransform
from ...preprocessing import Resample


class RandomAnisotropy(RandomTransform):
    """.

    Args:
        axes: Axis or tuple of axes along which the image .
        factor: .
        image_interpolation:
        resample_back:
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.

    .. note:: If the input image is 2D, all axes should be in ``(0, 1)``.
    """

    def __init__(
            self,
            axes: Union[int, Tuple[int, ...]] = 0,
            factor: float = 3,
            image_interpolation: str = 'nearest',
            resample_back: bool = False,
            p: float = 1,
            seed: Optional[int] = None,
            ):
        super().__init__(p=p, seed=seed)
        self.axes = self.parse_axes(axes)

    def apply_transform(self, sample: Subject) -> Subject:
        axes_to_flip_hot = self.get_params(self.axes, self.flip_probability)
        random_parameters_dict = {'axes': axes_to_flip_hot}
        items = sample.get_images_dict(intensity_only=False).items()

        source_spacing = sample.spacing
        transform = Resample(
            target_spacing,
            image_interpolation=self.image_interpolation,
        )
        transformed = transform(sample)
        if self.resample_back:
            transform = Resample(
                source_spacing,
                image_interpolation=self.image_interpolation,
            )
            transformed = transform(transformed)

        sample.add_transform(self, random_parameters_dict)
        return sample

    @staticmethod
    def get_params(axes: Tuple[int, ...], probability: float) -> List[bool]:
        axes_hot = [False, False, False]
        for axis in axes:
            random_number = torch.rand(1)
            flip_this = bool(probability > random_number)
            axes_hot[axis] = flip_this
        return axes_hot

    @staticmethod
    def parse_axes(axes: Union[int, Tuple[int, ...]]):
        axes_tuple = to_tuple(axes)
        for axis in axes_tuple:
            is_int = isinstance(axis, int)
            if not is_int or axis not in (0, 1, 2):
                raise ValueError('All axes must be 0, 1 or 2')
        return axes_tuple
