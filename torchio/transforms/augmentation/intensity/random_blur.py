from typing import Union, Tuple, Optional, List
import torch
import numpy as np
import scipy.ndimage as ndi
from ....torchio import DATA, TypeData, TypeTripletFloat, TypeSextetFloat
from ....data.subject import Subject
from ... import IntensityTransform
from .. import RandomTransform


class RandomBlur(RandomTransform, IntensityTransform):
    r"""Blur an image using a random-sized Gaussian filter.

    Args:
        std: Tuple :math:`(a_1, b_1, a_2, b_2, a_3, b_3)` representing the
            ranges (in mm) of the standard deviations
            :math:`(\sigma_1, \sigma_2, \sigma_3)` of the Gaussian kernels used
            to blur the image along each axis, where
            :math:`\sigma_i \sim \mathcal{U}(a_i, b_i)`.
            If two values :math:`(a, b)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(a, b)`.
            If only one value :math:`x` is provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x)`.
            If three values :math:`(x_1, x_2, x_3)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x_i)`.
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
        self.std_ranges = self.parse_params(std, None, 'std', min_constraint=0)

    def apply_transform(self, subject: Subject) -> Subject:
        random_parameters_images_dict = {}
        for image_name, image in self.get_images_dict(subject).items():
            transformed_tensors = []
            for channel_idx, tensor in enumerate(image[DATA]):
                std = self.get_params(self.std_ranges)
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
        subject.add_transform(self, random_parameters_images_dict)
        return subject

    def get_params(self, std_ranges: TypeSextetFloat) -> TypeTripletFloat:
        std = self.sample_uniform_sextet(std_ranges)
        return std


def blur(
        data: TypeData,
        spacing: TypeTripletFloat,
        std_voxel: TypeTripletFloat,
        ) -> torch.Tensor:
    assert data.ndim == 3
    std_physical = np.array(std_voxel) / np.array(spacing)
    blurred = ndi.gaussian_filter(data, std_physical)
    tensor = torch.from_numpy(blurred)
    return tensor
