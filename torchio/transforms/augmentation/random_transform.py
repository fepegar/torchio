"""
This is the docstring of random transform module
"""

from typing import Optional, Tuple, List

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
        keys: See :py:class:`~torchio.transforms.Transform`.
    """
    def __init__(
            self,
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, keys=keys)
        self._seed = seed

    def __call__(self, subject: Subject):
        self.check_seed()
        return super().__call__(subject)

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
    def sample_uniform(a, b):
        return torch.FloatTensor(1).uniform_(a, b)

    def sample_uniform_sextet(self, params):
        results = []
        for (a, b) in zip(params[::2], params[1::2]):
            results.append(self.sample_uniform(a, b))
        return torch.Tensor(results)

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
