from collections import defaultdict
from typing import Tuple

import torch

from ....utils import to_tuple
from ....typing import TypeRangeFloat
from ....data.subject import Subject
from ... import IntensityTransform
from .. import RandomTransform


class RandomGamma(RandomTransform, IntensityTransform):
    r"""Randomly change contrast of an image by raising its values to the power
    :math:`\gamma`.

    Args:
        log_gamma: Tuple :math:`(a, b)` to compute the exponent
            :math:`\gamma = e ^ \beta`,
            where :math:`\beta \sim \mathcal{U}(a, b)`.
            If a single value :math:`d` is provided, then
            :math:`\beta \sim \mathcal{U}(-d, d)`.
            Negative and positive values for this argument perform gamma
            compression and expansion, respectively.
            See the `Gamma correction`_ Wikipedia entry for more information.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. _Gamma correction: https://en.wikipedia.org/wiki/Gamma_correction

    .. note:: Fractional exponentiation of negative values is generally not
        well-defined for non-complex numbers.
        If negative values are found in the input image :math:`I`,
        the applied transform is :math:`\text{sign}(I) |I|^\gamma`,
        instead of the usual :math:`I^\gamma`. The
        :class:`~torchio.transforms.RescaleIntensity`
        transform may be used to ensure that all values are positive. This is
        generally not problematic, but it is recommended to visualize results
        on image with negative values. More information can be found on
        `this StackExchange question`_.

        .. _this StackExchange question: https://math.stackexchange.com/questions/317528/how-do-you-compute-negative-numbers-to-fractional-powers



    Example:
        >>> import torchio as tio
        >>> subject = tio.datasets.FPG()
        >>> transform = tio.RandomGamma(log_gamma=(-0.3, 0.3))  # gamma between 0.74 and 1.34
        >>> transformed = transform(subject)
    """  # noqa: E501
    def __init__(
            self,
            log_gamma: TypeRangeFloat = (-0.3, 0.3),
            **kwargs
            ):
        super().__init__(**kwargs)
        self.log_gamma_range = self._parse_range(log_gamma, 'log_gamma')

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        for name, image in self.get_images_dict(subject).items():
            gammas = [
                self.get_params(self.log_gamma_range)
                for _ in image.data
            ]
            arguments['gamma'][name] = gammas
        transform = Gamma(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(self, log_gamma_range: Tuple[float, float]) -> float:
        gamma = self.sample_uniform(*log_gamma_range).exp().item()
        return gamma


class Gamma(IntensityTransform):
    r"""Change contrast of an image by raising its values to the power
    :math:`\gamma`.

    Args:
        gamma: Exponent to which values in the image will be raised.
            Negative and positive values for this argument perform gamma
            compression and expansion, respectively.
            See the `Gamma correction`_ Wikipedia entry for more information.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. _Gamma correction: https://en.wikipedia.org/wiki/Gamma_correction

    .. note:: Fractional exponentiation of negative values is generally not
        well-defined for non-complex numbers.
        If negative values are found in the input image :math:`I`,
        the applied transform is :math:`\text{sign}(I) |I|^\gamma`,
        instead of the usual :math:`I^\gamma`. The
        :class:`~torchio.transforms.preprocessing.intensity.rescale.RescaleIntensity`
        transform may be used to ensure that all values are positive. This is
        generally not problematic, but it is recommended to visualize results
        on image with negative values. More information can be found on
        `this StackExchange question`_.

        .. _this StackExchange question: https://math.stackexchange.com/questions/317528/how-do-you-compute-negative-numbers-to-fractional-powers

    Example:
        >>> import torchio as tio
        >>> subject = tio.datasets.FPG()
        >>> transform = tio.Gamma(0.8)
        >>> transformed = transform(subject)
    """  # noqa: E501
    def __init__(
            self,
            gamma: float,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.args_names = ('gamma',)
        self.invert_transform = False

    def apply_transform(self, subject: Subject) -> Subject:
        gamma = self.gamma
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                gamma = self.gamma[name]
            gammas = to_tuple(gamma, length=len(image.data))
            transformed_tensors = []
            image.set_data(image.data.float())
            for gamma, tensor in zip(gammas, image.data):
                if self.invert_transform:
                    correction = power(tensor, 1 - gamma)
                    transformed_tensor = tensor * correction
                else:
                    transformed_tensor = power(tensor, gamma)
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject


def power(tensor, gamma):
    if tensor.min() < 0:
        output = tensor.sign() * tensor.abs() ** gamma
    else:
        output = tensor ** gamma
    return output
