from __future__ import annotations

from typing import Tuple

import torch

from .. import Transform
from ...typing import TypeRangeFloat


class RandomTransform(Transform):
    """Base class for stochastic augmentation transforms.

    Args:
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_include_exclude(self, kwargs):
        kwargs['include'] = self.include
        kwargs['exclude'] = self.exclude
        return kwargs

    def parse_degrees(
        self,
        degrees: TypeRangeFloat,
    ) -> Tuple[float, float]:
        return self._parse_range(degrees, 'degrees')

    def parse_translation(
        self,
        translation: TypeRangeFloat,
    ) -> Tuple[float, float]:
        return self._parse_range(translation, 'translation')

    @staticmethod
    def sample_uniform(a: float, b: float) -> float:
        return torch.FloatTensor(1).uniform_(a, b).item()

    @staticmethod
    def _get_random_seed() -> int:
        """Generate a random seed.

        Returns:
            A random seed as an int.
        """
        return int(torch.randint(0, 2**31, (1,)).item())

    def sample_uniform_sextet(self, params):
        results = []
        for a, b in zip(params[::2], params[1::2]):
            results.append(self.sample_uniform(a, b))
        return torch.Tensor(results)
