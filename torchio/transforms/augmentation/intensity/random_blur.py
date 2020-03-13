from typing import Tuple, Optional
import torch
import numpy as np
import SimpleITK as sitk
from ....utils import is_image_dict, nib_to_sitk, sitk_to_nib
from ....torchio import DATA, AFFINE, TYPE, INTENSITY, TypeData
from .. import RandomTransform


class RandomBlur(RandomTransform):
    """Blur an image using a random-sized Gaussian filter.

    Args:
        std_range:
        seed:
    """
    def __init__(
            self,
            std_range: Tuple[float, float] = (0, 4),
            seed: Optional[int] = None,
            ):
        super().__init__(seed=seed)
        self.std_range = std_range

    def apply_transform(self, sample: dict) -> dict:
        std = self.get_params(self.std_range)
        sample['random_blur'] = std
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            if image_dict[TYPE] != INTENSITY:
                continue
            image_dict[DATA][0] = blur(
                image_dict[DATA][0],
                image_dict[AFFINE],
                std,
            )
        return sample

    @staticmethod
    def get_params(std_range: Tuple[float, float]) -> np.ndarray:
        std = torch.FloatTensor(3).uniform_(*std_range).numpy()
        return std


def blur(data: TypeData, affine: TypeData, std: np.ndarray) -> torch.Tensor:
    image = nib_to_sitk(data, affine)
    image = sitk.DiscreteGaussian(image, std.tolist())
    array, _ = sitk_to_nib(image)
    tensor = torch.from_numpy(array)
    return tensor
