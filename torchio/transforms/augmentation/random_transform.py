"""
This is the docstring of random transform module
"""

from typing import Optional, Tuple, List

import torch
import numpy as np

from ...utils import gen_seed
from ... import TypeRangeFloat
from .. import Transform, TypeTransformInput


class RandomTransform(Transform):
    """Base class for stochastic augmentation transforms.

    Args:
        p: Probability that this transform will be applied.
        keys: See :py:class:`~torchio.transforms.Transform`.
    """
    def __init__(
            self,
            p: float = 1,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, keys=keys)

    def __call__(
        self,
        data: TypeTransformInput,
        seed: int = None,
    ) -> TypeTransformInput:
        """Transform data and return a result of the same type.

        Args:
            data: Instance of :py:class:`~torchio.Subject`, 4D
                :py:class:`torch.Tensor` or 4D NumPy array with dimensions
                :math:`(C, W, H, D)`, where :math:`C` is the number of channels
                and :math:`W, H, D` are the spatial dimensions. If the input is
                a tensor, the affine matrix is an identity and a tensor will be
                also returned.
            seed: Seed for :py:mod:`torch` random number generator.
        """
        if not seed:
            seed = gen_seed()

        # Store the current rng_state to reset it after the execution
        torch_rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed=seed)
        self.seed = seed

        transformed = super().__call__(data=data)

        torch.random.set_rng_state(torch_rng_state)
        return transformed

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
