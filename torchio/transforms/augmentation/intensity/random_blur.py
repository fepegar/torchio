from typing import Union, Tuple, Optional, List
import torch
import numpy as np
import SimpleITK as sitk
from ....utils import nib_to_sitk, sitk_to_nib
from ....torchio import DATA, AFFINE, TypeData
from ....data.subject import Subject
from .. import RandomTransform


class RandomBlur(RandomTransform):
    r"""Blur an image using a random-sized Gaussian filter.

    Args:
        std: Tuple :math:`(a, b)` to compute the standard deviations
            :math:`(\sigma_1, \sigma_2, \sigma_3)` of the Gaussian kernels used
            to blur the image along each axis,
            where :math:`\sigma_i \sim \mathcal{U}(a, b)` mm.
            If only one value :math:`d` is provided,
            :math:`\sigma_i \sim \mathcal{U}(0, d)`.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
        keys: See :py:class:`~torchio.transforms.Transform`.
    """
    def __init__(
            self,
            std: Union[float, Tuple[float, float]] = (0, 4),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.std_range = self.parse_range(std, 'std', min_constraint=0)

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        for image_name, image in sample.get_images_dict().items():
            transformed_tensors = []
            for channel_idx, tensor in enumerate(image[DATA]):
                std = self.get_params(self.std_range)
                random_parameters_dict = {'std': std}
                key = f'{image_name}_channel_{channel_idx}'
                random_parameters_images_dict[key] = random_parameters_dict
                transformed_tensor = blur(
                    tensor,
                    image[AFFINE],
                    std,
                )
                transformed_tensors.append(transformed_tensor)
            image[DATA] = torch.stack(transformed_tensors)
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    @staticmethod
    def get_params(std_range: Tuple[float, float]) -> np.ndarray:
        std = torch.FloatTensor(3).uniform_(*std_range).numpy()
        return std


def blur(data: TypeData, affine: TypeData, std: np.ndarray) -> torch.Tensor:
    assert data.ndim == 3
    image = nib_to_sitk(data[np.newaxis], affine)
    image = sitk.DiscreteGaussian(image, std.tolist())
    array, _ = sitk_to_nib(image)
    tensor = torch.from_numpy(array[0])
    return tensor
