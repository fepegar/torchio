from typing import Tuple, Optional
import torch
import numpy as np
from ....torchio import DATA
from ....data.subject import Subject
from .. import RandomTransform


class RandomNoise(RandomTransform):
    r"""Add random Gaussian noise.

    Args:
        std: Standard deviation :math:`\sigma` of the Gaussian distribution
            from which the noise is sampled.
            If two values :math:`(a, b)` are providede,
            then :math:`\sigma \sim \mathcal{U}(a, b)`.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
    """
    def __init__(
            self,
            std: Tuple[float, float] = (0, 0.25),
            p: float = 1,
            seed: Optional[int] = None,
            ):
        super().__init__(p=p, seed=seed)
        self.std_range = self.parse_range(std, 'std')
        if any(np.array(self.std_range) < 0):
            message = (
                'Standard deviation std must greater or equal to zero,'
                f' not "{self.std_range}"'
            )
            raise ValueError(message)

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        for image_name, image_dict in sample.get_images_dict().items():
            std = self.get_params(self.std_range)
            random_parameters_dict = {'std': std}
            random_parameters_images_dict[image_name] = random_parameters_dict
            image_dict[DATA] = add_noise(image_dict[DATA], std)
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    @staticmethod
    def get_params(std_range: Tuple[float, float]) -> float:
        std = torch.FloatTensor(1).uniform_(*std_range).item()
        return std


def add_noise(tensor: torch.Tensor, std: float) -> torch.Tensor:
    noise = torch.FloatTensor(*tensor.shape).normal_(mean=0, std=std)
    tensor = tensor + noise
    return tensor
