import warnings
from collections import defaultdict
from typing import Tuple, Optional, Sequence

import torch

from ....utils import to_tuple
from ....torchio import DATA, TypeRangeFloat
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
        p: Probability that this transform will be applied.
        keys: See :class:`~torchio.transforms.Transform`.

    .. _Gamma correction: https://en.wikipedia.org/wiki/Gamma_correction

    .. warning:: Fractional exponentiation of negative values is generally not
        well-defined for non-complex numbers.
        If negative values are found in the input image :math:`I`,
        the applied transform is :math:`\text{sign}(I) |I|^\gamma`,
        instead of the usual :math:`I^\gamma`. The
        :class:`~torchio.transforms.preprocessing.intensity.rescale.RescaleIntensity`
        transform may be used to ensure that all values are positive.

    Example:
        >>> import torchio as tio
        >>> subject = tio.datasets.FPG()
        >>> transform = tio.RandomGamma(log_gamma=(-0.3, 0.3))  # gamma between 0.74 and 1.34
        >>> transformed = transform(subject)
    """
    def __init__(
            self,
            log_gamma: TypeRangeFloat = (-0.3, 0.3),
            p: float = 1,
            keys: Optional[Sequence[str]] = None,
            ):
        super().__init__(p=p, keys=keys)
        self.log_gamma_range = self.parse_range(log_gamma, 'log_gamma')

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        for name, image in self.get_images_dict(subject).items():
            gammas = [self.get_params(self.log_gamma_range) for _ in image.data]
            arguments['gamma'][name] = gammas
        transform = Gamma(**arguments)
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
        keys: See :class:`~torchio.transforms.Transform`.

    .. _Gamma correction: https://en.wikipedia.org/wiki/Gamma_correction

    .. warning:: Fractional exponentiation of negative values is generally not
        well-defined for non-complex numbers.
        If negative values are found in the input image :math:`I`,
        the applied transform is :math:`\text{sign}(I) |I|^\gamma`,
        instead of the usual :math:`I^\gamma`. The
        :class:`~torchio.transforms.preprocessing.intensity.rescale.RescaleIntensity`
        transform may be used to ensure that all values are positive.

    Example:
        >>> import torchio as tio
        >>> subject = tio.datasets.FPG()
        >>> transform = tio.Gamma(0.8)
        >>> transformed = transform(subject)
    """
    def __init__(
            self,
            gamma: float,
            keys: Optional[Sequence[str]] = None,
            ):
        super().__init__(keys=keys)
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
            for gamma, tensor in zip(gammas, image.data):
                if self.invert_transform:
                    correction = power(tensor, 1 - gamma)
                    transformed_tensor = tensor * correction
                else:
                    transformed_tensor = power(tensor, gamma)
                transformed_tensors.append(transformed_tensor)
            image[DATA] = torch.stack(transformed_tensors)
        return subject


def power(tensor, gamma):
    if tensor.min() < 0:
        message = (
            'Negative values found in input tensor. See the documentation for'
            ' more details on the implemented workaround:'
            ' https://torchio.readthedocs.io/transforms/augmentation.html#randomgamma'
        )
        warnings.warn(message, RuntimeWarning)
        output = tensor.sign() * tensor.abs() ** gamma
    else:
        output = tensor ** gamma
    return output
