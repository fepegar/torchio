from collections import defaultdict
from collections.abc import Sequence
from typing import Union

import torch

from ....data.subject import Subject
from ...intensity_transform import IntensityTransform
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
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(
        self,
        mean: Union[float, tuple[float, float]] = 0,
        std: Union[float, tuple[float, float]] = (0, 0.25),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mean_range = self._parse_range(mean, 'mean')
        self.std_range = self._parse_range(std, 'std', min_constraint=0)

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict = self.get_images_dict(subject)
        if not images_dict:
            return subject

        arguments: dict[str, dict] = defaultdict(dict)
        for image_name in images_dict:
            mean, std, seed = self.get_params(self.mean_range, self.std_range)
            arguments['mean'][image_name] = mean
            arguments['std'][image_name] = std
            arguments['seed'][image_name] = seed
        transform = Noise(**self.add_base_args(arguments))
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed

    def get_params(
        self,
        mean_range: tuple[float, float],
        std_range: tuple[float, float],
    ) -> tuple[float, float, int]:
        mean = self.sample_uniform(*mean_range)
        std = self.sample_uniform(*std_range)
        seed = self._get_random_seed()
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
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(
        self,
        mean: Union[float, dict[str, float]],
        std: Union[float, dict[str, float]],
        seed: Union[int, Sequence[int]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mean = mean  # type: ignore[assignment]
        self.std = std
        self.seed = seed
        self.invert_transform = False
        self.args_names = ['mean', 'std', 'seed']

    def apply_transform(self, subject: Subject) -> Subject:
        mean, std, seed = args = self.mean, self.std, self.seed
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                values = (arg[name] for arg in args)  # type: ignore[index,call-overload]
                mean, std, seed = values  # type: ignore[assignment]
            with self._use_seed(seed):
                assert isinstance(mean, float)
                assert isinstance(std, float)
                noise = get_noise(image.data, mean, std)
            if self.invert_transform:
                noise *= -1
            image.set_data(image.data + noise)
        return subject


def get_noise(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return torch.randn(*tensor.shape) * std + mean
