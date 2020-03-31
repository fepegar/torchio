from typing import Tuple, Optional
import torch
import numpy as np
from ....utils import is_image_dict
from ....torchio import DATA, TYPE, INTENSITY, TypeData
from .. import RandomTransform


class RandomNoise(RandomTransform):
    r"""Add random Gaussian noise.

    Args:
        std: Standard deviation :math:`\sigma` of the Gaussian distribution
            from which the noise is sampled.
            If two values :math:`(a, b)` are providede,
            then :math:`\sigma \sim \mathcal{U}(a, b)`.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
    """
    def __init__(
            self,
            std: Tuple[float, float] = (0, 0.25),
            seed: Optional[int] = None,
            ):
        super().__init__(seed=seed)
        self.std_range = self.parse_range(std, 'std')
        if any(np.array(self.std_range) < 0):
            message = (
                'Standard deviation std must greater or equal to zero,'
                f' not "{self.std_range}"'
            )
            raise ValueError(message)

    def apply_transform(self, sample: dict) -> dict:
        std = self.get_params(self.std_range)
        sample['random_noise'] = std
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            if image_dict[TYPE] != INTENSITY:
                continue
            image_dict[DATA] = add_noise(image_dict[DATA], std)
        return sample

    @staticmethod
    def get_params(std_range: Tuple[float, float]) -> float:
        std = torch.FloatTensor(1).uniform_(*std_range).item()
        return std


def add_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    noise = torch.FloatTensor(*tensor.shape).normal_(mean=0, std=std)
    tensor = tensor + noise
    return tensor
