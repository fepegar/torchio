from collections import defaultdict
from typing import Tuple, Optional, Union, Dict, Sequence

import torch
from ....torchio import DATA
from ....data.subject import Subject
from ... import IntensityTransform
from .. import RandomTransform


class RandomNoise(RandomTransform, IntensityTransform):
    r"""Add Gaussian noise with random parameters.

    Add noise sampled from a normal distribution with random parameters.

    Args:
        mean: Mean :math:`\mu` of the Gaussian distribution
            from which the noise is sampled.
            If two values :math:`(a, b)` are provided,
            then :math:`\mu \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`\mu \sim \mathcal{U}(-d, d)`.
        std: Standard deviation :math:`\sigma` of the Gaussian distribution
            from which the noise is sampled.
            If two values :math:`(a, b)` are provided,
            then :math:`\sigma \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`\sigma \sim \mathcal{U}(0, d)`.
        p: Probability that this transform will be applied.
        keys: See :class:`~torchio.transforms.Transform`.
    """
    def __init__(
            self,
            mean: Union[float, Tuple[float, float]] = 0,
            std: Union[float, Tuple[float, float]] = (0, 0.25),
            p: float = 1,
            keys: Optional[Sequence[str]] = None,
            ):
        super().__init__(p=p, keys=keys)
        self.mean_range = self.parse_range(mean, 'mean')
        self.std_range = self.parse_range(std, 'std', min_constraint=0)

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        for image_name in self.get_images_dict(subject):
            mean, std, seed = self.get_params(self.mean_range, self.std_range)
            arguments['mean'][image_name] = mean
            arguments['std'][image_name] = std
            arguments['seed'][image_name] = seed
        transform = Noise(**arguments)
        transformed = transform(subject)
        return transformed

    def get_params(
            self,
            mean_range: Tuple[float, float],
            std_range: Tuple[float, float],
            ) -> Tuple[float, float]:
        mean = self.sample_uniform(*mean_range).item()
        std = self.sample_uniform(*std_range).item()
        seed = self.get_random_seed()
        return mean, std, seed


class Noise(IntensityTransform):
    r"""Add Gaussian noise.

    Add noise sampled from a normal distribution.

    Args:
        mean: Mean :math:`\mu` of the Gaussian distribution
            from which the noise is sampled.
        std: Standard deviation :math:`\sigma` of the Gaussian distribution
            from which the noise is sampled.
        seed: Seed for the random number generator.
        keys: See :class:`~torchio.transforms.Transform`.
    """
    def __init__(
            self,
            mean: Union[float, Dict[str, float]],
            std: Union[float, Dict[str, float]],
            seed: Union[int, Sequence[int]],
            keys: Optional[Sequence[str]] = None,
            ):
        super().__init__(keys=keys)
        self.mean = mean
        self.std = std
        self.seed = seed
        self.invert_transform = False
        self.args_names = 'mean', 'std', 'seed'

    def apply_transform(self, subject: Subject) -> Subject:
        args = self.mean, self.std, self.seed
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                mean, std, seed = [arg[name] for arg in args]
            with self._use_seed(seed):
                noise = get_noise(image[DATA], mean, std)
            if self.invert_transform:
                noise *= -1
            image[DATA] = image[DATA] + noise
        return subject


def get_noise(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return torch.randn(*tensor.shape) * std + mean
