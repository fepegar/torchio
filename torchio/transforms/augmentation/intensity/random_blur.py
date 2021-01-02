from collections import defaultdict
from typing import Union, Tuple, Dict

import torch
import numpy as np
import scipy.ndimage as ndi

from ....utils import to_tuple
from ....typing import TypeData, TypeTripletFloat, TypeSextetFloat
from ....data.subject import Subject
from ... import IntensityTransform
from .. import RandomTransform


class RandomBlur(RandomTransform, IntensityTransform):
    r"""Blur an image using a random-sized Gaussian filter.

    Args:
        std: Tuple :math:`(a_1, b_1, a_2, b_2, a_3, b_3)` representing the
            ranges (in mm) of the standard deviations
            :math:`(\sigma_1, \sigma_2, \sigma_3)` of the Gaussian kernels used
            to blur the image along each axis, where
            :math:`\sigma_i \sim \mathcal{U}(a_i, b_i)`.
            If two values :math:`(a, b)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(a, b)`.
            If only one value :math:`x` is provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x)`.
            If three values :math:`(x_1, x_2, x_3)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x_i)`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            std: Union[float, Tuple[float, float]] = (0, 2),
            **kwargs
            ):
        super().__init__(**kwargs)
        self.std_ranges = self.parse_params(std, None, 'std', min_constraint=0)

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        for name, image in self.get_images_dict(subject).items():
            stds = [self.get_params(self.std_ranges) for _ in image.data]
            arguments['std'][name] = stds
        transform = Blur(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(self, std_ranges: TypeSextetFloat) -> TypeTripletFloat:
        std = self.sample_uniform_sextet(std_ranges)
        return std


class Blur(IntensityTransform):
    r"""Blur an image using a Gaussian filter.

    Args:
        std: Tuple :math:`(\sigma_1, \sigma_2, \sigma_3)` representing the
            the standard deviations (in mm) of the standard deviations
            of the Gaussian kernels used to blur the image along each axis.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            std: Union[TypeTripletFloat, Dict[str, TypeTripletFloat]],
            **kwargs
            ):
        super().__init__(**kwargs)
        self.std = std
        self.args_names = ('std',)

    def apply_transform(self, subject: Subject) -> Subject:
        std = self.std
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                std = self.std[name]
            stds = to_tuple(std, length=len(image.data))
            transformed_tensors = []
            for std, tensor in zip(stds, image.data):
                transformed_tensor = blur(
                    tensor,
                    image.spacing,
                    std,
                )
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject


def blur(
        data: TypeData,
        spacing: TypeTripletFloat,
        std_voxel: TypeTripletFloat,
        ) -> torch.Tensor:
    assert data.ndim == 3
    std_physical = np.array(std_voxel) / np.array(spacing)
    blurred = ndi.gaussian_filter(data, std_physical)
    tensor = torch.from_numpy(blurred)
    return tensor
