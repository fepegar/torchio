from typing import Tuple, Optional
import torch
from ....utils import is_image_dict
from ....torchio import DATA, INTENSITY, TypeData
from .. import RandomTransform


class RandomNoise(RandomTransform):
    def __init__(
            self,
            std_range: Tuple[float, float] = (0, 0.25),
            seed: Optional[int] = None,
            verbose: bool = False,
            ):
        super().__init__(seed=seed, verbose=verbose)
        self.std_range = std_range

    def apply_transform(self, sample: dict) -> dict:
        std = self.get_params(self.std_range)
        sample['random_noise'] = std
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] != INTENSITY:
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
