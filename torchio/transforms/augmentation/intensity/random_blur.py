from typing import Union, Tuple, Optional, List
import torch
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi
from ....torchio import DATA, AFFINE, TypeData, TypeTripletFloat
from ....data.subject import Subject
from ... import IntensityTransform
from .. import RandomTransform


class RandomBlur(RandomTransform, IntensityTransform):
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
            std: Union[float, Tuple[float, float]] = (0, 2),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.std_range = self.parse_range(std, 'std', min_constraint=0)

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        for image_name, image in self.get_images_dict(sample).items():
            transformed_tensors = []
            for channel_idx, tensor in enumerate(image[DATA]):
                std = self.get_params(self.std_range)
                random_parameters_dict = {'std': std}
                key = f'{image_name}_channel_{channel_idx}'
                random_parameters_images_dict[key] = random_parameters_dict
                transformed_tensor = blur(
                    tensor,
                    image.spacing,
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


def blur(
        data: TypeData,
        spacing: TypeTripletFloat,
        std_voxel: np.ndarray,
        ) -> torch.Tensor:
    assert data.ndim == 3
    std_physical = np.array(std_voxel) / np.array(spacing)
    blurred = ndi.gaussian_filter(data, std_physical)
    tensor = torch.from_numpy(blurred)
    return tensor
