from typing import Union, Tuple, Optional
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
            If a single value :math:`n` is provided, then :math:`a = b = n`.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
    """
    def __init__(
            self,
            std: Union[float, Tuple[float, float]] = (0, 4),
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
            image_dict[DATA][0] = blur(
                image_dict[DATA][0],
                image_dict[AFFINE],
                std,
            )
        sample.add_transform(self, random_parameters_images_dict)
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
