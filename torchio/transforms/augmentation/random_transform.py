"""
This is the docstring of random transform module
"""

from typing import Optional, Tuple, Sequence

import torch

from ... import TypeRangeFloat
from .. import Transform


class RandomTransform(Transform):
    """Base class for stochastic augmentation transforms.

    Args:
        p: Probability that this transform will be applied.
        keys: See :class:`~torchio.transforms.Transform`.
    """
    def __init__(
            self,
            p: float = 1,
            keys: Optional[Sequence[str]] = None,
            ):
        super().__init__(p=p, keys=keys)

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

    @staticmethod
    def get_random_seed():
        """Generate a random seed.

        Returns:
            A random seed as an int.
        """
        return torch.randint(0, 2**31, (1,)).item()

    def sample_uniform_sextet(self, params):
        results = []
        for (a, b) in zip(params[::2], params[1::2]):
            results.append(self.sample_uniform(a, b))
        return torch.Tensor(results)
