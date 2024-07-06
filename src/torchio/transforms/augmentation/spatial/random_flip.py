from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch

from .. import RandomTransform
from ... import SpatialTransform
from ....data.subject import Subject


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
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> fpg = tio.datasets.FPG()
        >>> flip = tio.RandomFlip(axes=('LR',))  # flip along lateral axis only

    .. tip:: It is handy to specify the axes as anatomical labels when the
        image orientation is not known.
    """

    def __init__(
        self,
        axes: Union[int, Tuple[int, ...]] = 0,
        flip_probability: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axes = self.parse_axes(axes)
        self.flip_probability = self.parse_probability(flip_probability)

    def apply_transform(self, subject: Subject) -> Subject:
        potential_axes = self.ensure_axes_indices(subject, self.axes)
        axes_to_flip_hot = self.get_params(self.flip_probability)
        for i in range(3):
            if i not in potential_axes:
                axes_to_flip_hot[i] = False
        (axes,) = np.where(axes_to_flip_hot)
        axes = axes.tolist()
        if not axes:
            return subject

        arguments = {'axes': axes}
        transform = Flip(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed

    @staticmethod
    def get_params(probability: float) -> List[bool]:
        return (probability > torch.rand(3)).tolist()


class Flip(SpatialTransform):
    """Reverse the order of elements in an image along the given axes.

    Args:
        axes: Index or tuple of indices of the spatial dimensions along which
            the image will be flipped. See
            :class:`~torchio.transforms.augmentation.spatial.random_flip.RandomFlip`
            for more information.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. tip:: It is handy to specify the axes as anatomical labels when the
        image orientation is not known.
    """

    def __init__(
        self, axes: Union[int, Tuple[int, ...], str, Tuple[str, ...]], **kwargs
    ):
        super().__init__(**kwargs)
        self.axes = self.parse_axes(axes)
        self.args_names = ('axes',)

    def apply_transform(self, subject: Subject) -> Subject:
        axes = self.ensure_axes_indices(subject, self.axes)
        for image in self.get_images(subject):
            self.flip_image(image, axes)
        return subject

    @staticmethod
    def flip_image(image, axes):
        spatial_axes = np.array(axes, int) + 1
        data = image.numpy()
        data = np.flip(data, axis=spatial_axes)
        data = data.copy()  # remove negative strides
        data = torch.as_tensor(data)
        image.set_data(data)

    @staticmethod
    def is_invertible():
        return True

    def inverse(self):
        return self
