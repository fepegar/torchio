from typing import Union, Tuple, Optional, List
import torch
from ....torchio import DATA
from ....utils import is_image_dict, to_tuple
from .. import RandomTransform


class RandomFlip(RandomTransform):
    """Reverse the order of elements in an image along the given axes.

    Args:
        axes: Axis or tuple of axes along which the image will be flipped.
        flip_probability: Probability that the image will be flipped. This is
            computed on a per-axis basis.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
    """

    def __init__(
            self,
            axes: Union[int, Tuple[int, ...]] = 0,
            flip_probability: float = 0.5,
            seed: Optional[int] = None,
            ):
        super().__init__(seed=seed)
        self.axes = self.parse_axes(axes)
        self.flip_probability = self.parse_probability(
            flip_probability,
            'flip_probability',
        )

    def apply_transform(self, sample: dict) -> dict:
        axes_to_flip_hot = self.get_params(self.axes, self.flip_probability)
        sample['random_flip'] = axes_to_flip_hot
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            tensor = image_dict[DATA]
            dims = []
            for dim, flip_this in enumerate(axes_to_flip_hot):
                if not flip_this:
                    continue
                actual_dim = dim + 1  # images are 4D
                dims.append(actual_dim)
            tensor = torch.flip(tensor, dims=dims)
            image_dict[DATA] = tensor
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
