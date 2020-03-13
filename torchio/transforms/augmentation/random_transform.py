"""
This is the docstring of random transform module
"""

import numbers
from typing import Optional, Tuple, Union
from abc import abstractmethod
import torch
import numpy as np
from .. import Transform, Interpolation
from ... import TypeNumber, TypeRangeFloat


class RandomTransform(Transform):
    """Base class for stochastic augmentation transforms.

    Args:
        seed: Seed for :mod:`torch` random number generator.
    """
    def __init__(
            self,
            seed: Optional[int] = None,
            ):
        super().__init__()
        self._seed = seed

    def __call__(self, sample: dict):
        self.check_seed()
        return super().__call__(sample)

    @staticmethod
    def parse_range(
            nums_range: Union[TypeNumber, Tuple[TypeNumber, TypeNumber]],
            name: str,
            ) -> Tuple[TypeNumber, TypeNumber]:
        r"""Adapted from ``torchvision.transforms.RandomRotation``.

        Args:
            nums_range: Tuple of two numbers :math:`(n_{min}, n_{max})`,
                where :math:`n_{min} \leq n_{max}`.
                If a single positive number :math:`n` is provided,
                :math:`n_{min} = -n` and :math:`n_{max} = n`.
            name: Name of the parameter, so that an informative error message
                can be printed.

        Returns:
            A tuple of two numbers :math:`(n_{min}, n_{max})`.

        Raises:
            ValueError: if :attr:`nums_range` is negative
            ValueError: if :math:`n_{max} \lt n_{min}`.
        """
        if isinstance(nums_range, numbers.Number):
            if nums_range < 0:
                raise ValueError(
                    f'If {name} is a single number,'
                    f' it must be positive, not {nums_range}')
            return (-nums_range, nums_range)
        else:
            if len(nums_range) != 2:
                raise ValueError(
                    f'If {name} is a sequence,'
                    f' it must be of len 2, not {nums_range}')
            min_degree, max_degree = nums_range
            if min_degree > max_degree:
                raise ValueError(
                    f'If {name} is a sequence, the second value must be'
                    f' equal or greater than the first, not {nums_range}')
            return nums_range

    def parse_degrees(
            self,
            degrees: TypeRangeFloat,
            ) -> Tuple[float, float]:
        return self.parse_range(degrees, 'degrees')

    def parse_translation(
            self,
            translation: TypeRangeFloat,
            ) -> Tuple[float, float]:
        return self.parse_range(translation, 'translation')

    @staticmethod
    def parse_probability(p: float, name: str) -> float:
        if not (isinstance(p, numbers.Number) and 0 <= p <= 1):
            raise ValueError(f'{name} must be a number in [0, 1]')
        return p

    @staticmethod
    def parse_interpolation(interpolation: Interpolation) -> Interpolation:
        if not isinstance(interpolation, Interpolation):
            message = (
                'image_interpolation must be'
                ' a member of torchio.Interpolation'
            )
            raise TypeError(message)
        return interpolation

    def check_seed(self) -> None:
        if self._seed is not None:
            torch.manual_seed(self._seed)

    @staticmethod
    def fourier_transform(array: np.ndarray):
        transformed = np.fft.fftn(array)
        fshift = np.fft.fftshift(transformed)
        return fshift

    @staticmethod
    def inv_fourier_transform(fshift: np.ndarray):
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifftn(f_ishift)
        return np.abs(img_back)
