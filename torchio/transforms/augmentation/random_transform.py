import numbers
from typing import Optional, Tuple, Union
from abc import abstractmethod
import torch
from .. import Transform, Interpolation
from ... import TypeNumber


class RandomTransform(Transform):
    def __init__(self, seed: Optional[int] = None, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.seed = seed

    def __call__(self, sample: dict):
        self.check_seed()
        return super().__call__(sample)

    @staticmethod
    @abstractmethod
    def get_params(*args, **kwargs):
        pass

    @staticmethod
    def parse_range(
            nums_range: Union[TypeNumber, Tuple[TypeNumber]],
            name: str,
            ) -> Tuple[TypeNumber, TypeNumber]:
        """Adapted from torchvision.RandomRotation"""
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
            degrees: Union[float, Tuple[float]],
            ) -> Tuple[float, float]:
        return self.parse_range(degrees, 'degrees')

    def parse_translation(
            self,
            translation: Union[float, Tuple[float]],
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
        if self.seed is not None:
            if self.verbose:
                print('Setting torch seed to', self.seed)
            torch.manual_seed(self.seed)
