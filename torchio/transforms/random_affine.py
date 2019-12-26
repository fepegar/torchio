import numbers
import torch
import numpy as np
import SimpleITK as sitk
from ..torchio import LABEL
from ..utils import is_image_dict
from .interpolation import Interpolation
from .random_transform import RandomTransform


class RandomAffine(RandomTransform):
    def __init__(
            self,
            scales=(0.9, 1.1),
            degrees=10,
            isotropic=False,
            image_interpolation=Interpolation.LINEAR,
            seed=None,
            verbose=False,
            ):
        super().__init__(seed=seed, verbose=verbose)
        self.scales = scales
        self.degrees = self.parse_degrees(degrees)
        self.isotropic = isotropic
        self.image_interpolation = image_interpolation

    @staticmethod
    def parse_degrees(degrees):
        """Adapted from torchvision.RandomRotation"""
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    'If degrees is a single number,'
                    f' it must be positive, not {degrees}')
            return (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    'If degrees is a sequence,'
                    f' it must be of len 2, not {degrees}')
            min_degree, max_degree = degrees
            if min_degree > max_degree:
                raise ValueError(
                    'If degrees is a sequence, the second value must be'
                    f' equal or greater than the first, not {degrees}')
            return degrees

    def apply_transform(self, sample):
        scaling_params, rotation_params = self.get_params(
            self.scales, self.degrees, self.isotropic)
        sample['random_scaling'] = scaling_params
        sample['random_rotation'] = rotation_params
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] == LABEL:
                interpolation = Interpolation.NEAREST
            else:
                interpolation = self.image_interpolation
            image_dict['data'] = self.apply_affine_transform(
                image_dict['data'],
                image_dict['affine'],
                scaling_params,
                rotation_params,
                interpolation,
            )
        return sample

    @staticmethod
    def get_params(scales, degrees, isotropic):
        scaling_params = torch.FloatTensor(3).uniform_(*scales).tolist()
        if isotropic:
            scaling_params = 3 * scaling_params[0]
        rotation_params = torch.FloatTensor(3).uniform_(*degrees).tolist()
        return scaling_params, rotation_params

    @staticmethod
    def get_scaling_transform(scaling_params):
        """
        scaling_params are inverted so that they are more intuitive
        For example, 1.5 means the objects look 1.5 times larger
        """
        transform = sitk.ScaleTransform(3)
        scaling_params = 1 / np.array(scaling_params)
        transform.SetScale(scaling_params)
        return transform

    @staticmethod
    def get_rotation_transform(rotation_params):
        """
        rotation_params is in degrees
        """
        transform = sitk.Euler3DTransform()
        rotation_params = np.radians(rotation_params)
        transform.SetRotation(*rotation_params)
        return transform

    def apply_affine_transform(
            self,
            array,
            affine,
            scaling_params,
            rotation_params,
            interpolation: Interpolation,
            ):
        if array.ndim != 4:
            message = (
                'Only 4D images (channels, i, j, k) are supported,'
                f' not {array.shape}'
            )
            raise NotImplementedError(message)
        for i, channel_array in enumerate(array):  # use sitk.VectorImage?
            image = self.nib_to_sitk(channel_array, affine)
            scaling_transform = self.get_scaling_transform(scaling_params)
            rotation_transform = self.get_rotation_transform(rotation_params)
            transform = sitk.Transform(3, sitk.sitkComposite)
            transform.AddTransform(scaling_transform)
            transform.AddTransform(rotation_transform)
            resampled = sitk.Resample(
                image,
                transform,
                interpolation.value,
            )
            channel_array = sitk.GetArrayFromImage(resampled)
            channel_array = channel_array.transpose(2, 1, 0)  # ITK to NumPy
            array[i] = channel_array
        return array
