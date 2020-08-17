from typing import Union, Tuple, Optional, List
import torch
import numpy as np
from ....torchio import DATA
from ....data.subject import Subject
from ....utils import to_tuple
from ... import SpatialTransform
from .. import RandomTransform


class RandomFlip(RandomTransform, SpatialTransform):
    """Reverse the order of elements in an image along the given axes.

    Args:
        axes: Index or tuple of indices of the spatial dimensions along which
            the image might be flipped. If they are integers, they must be in
            ``(0, 1, 2)``. Anatomical labels may also be used, such as
            ``'Left'``, ``'Right'``, ``'Anterior'``, ``'Posterior'``,
            ``'Inferior'``, ``'Superior'``, ``'Height'`` and ``'Width'``,
            ``'AP'`` (antero-posterior), ``'lr'`` (lateral), ``'w'`` (width) or
            ``'i'`` (inferior). Only the first letter of the string will be
            used. If the image is 2D, ``'Height'`` and ``'Width'`` may be
            used.
        flip_probability: Probability that the image will be flipped. This is
            computed on a per-axis basis.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
        keys: See :py:class:`~torchio.transforms.Transform`.

    Example:
        >>> import torchio as tio
        >>> fpg = tio.datasets.FPG()
        >>> flip = tio.RandomFlip(axes=('LR'))  # flip along lateral axis only

    .. tip:: It is handy to specify the axes as anatomical labels when the image
        orientation is not known.
    """

    def __init__(
            self,
            axes: Union[int, Tuple[int, ...]] = 0,
            flip_probability: float = 0.5,
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.axes = self.parse_axes(axes)
        self.flip_probability = self.parse_probability(
            flip_probability,
        )

    def apply_transform(self, sample: Subject) -> dict:
        axes = self.axes
        axes_to_flip_hot = self.get_params(self.flip_probability)
        if any(isinstance(n, str) for n in axes):
            sample.check_consistent_orientation()
            image = sample.get_first_image()
            axes = sorted([4 + image.axis_name_to_index(n) for n in axes])
        for i in range(3):
            if i not in axes:
                axes_to_flip_hot[i] = False
        random_parameters_dict = {'axes': axes_to_flip_hot}
        items = self.get_images_dict(sample).items()
        for image_name, image in items:
            dims = []
            for dim, flip_this in enumerate(axes_to_flip_hot):
                if not flip_this:
                    continue
                actual_dim = dim + 1  # images are 4D
                dims.append(actual_dim)
            if dims:
                data = image.numpy()
                data = np.flip(data, axis=dims)
                data = data.copy()  # remove negative strides
                data = torch.from_numpy(data)
                image[DATA] = data
        sample.add_transform(self, random_parameters_dict)
        return sample

    @staticmethod
    def get_params(probability: float) -> List[bool]:
        return (probability > torch.rand(3)).tolist()

    @staticmethod
    def parse_axes(axes: Union[int, Tuple[int, ...]]):
        axes_tuple = to_tuple(axes)
        for axis in axes_tuple:
            is_int = isinstance(axis, int)
            is_string = isinstance(axis, str)
            if not is_string and not (is_int and axis in (0, 1, 2)):
                message = f'All axes must be 0, 1 or 2, but found "{axis}"'
                raise ValueError(message)
        return axes_tuple
