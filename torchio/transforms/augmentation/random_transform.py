"""
This is the docstring of random transform module
"""

from typing import Optional, Tuple

import torch
import numpy as np

from ...data.subject import Subject
from ... import TypeRangeFloat
from .. import Transform


class RandomTransform(Transform):
    """Base class for stochastic augmentation transforms.

    Args:
        p: Probability that this transform will be applied.
        seed: Seed for :py:mod:`torch` random number generator.
    """
    def __init__(
            self,
            p: float = 1,
            seed: Optional[int] = None,
            ):
        super().__init__(p=p)
        self._seed = seed

    def __call__(self, sample: Subject):
        self.check_seed()
        return super().__call__(sample)

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
        return img_back
