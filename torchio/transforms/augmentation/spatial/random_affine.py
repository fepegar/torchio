from typing import Tuple, Optional, List
import torch
import numpy as np
import SimpleITK as sitk
from ....utils import is_image_dict, check_consistent_shape
from ....torchio import LABEL, DATA, AFFINE
from .. import Interpolation
from .. import RandomTransform


class RandomAffine(RandomTransform):
    def __init__(
            self,
            scales: Tuple[float, float] = (0.9, 1.1),
            degrees: float = 10,
            isotropic: bool = False,
            image_interpolation: Interpolation = Interpolation.LINEAR,
            seed: Optional[int] = None,
            verbose: bool = False,
            ):
        super().__init__(seed=seed, verbose=verbose)
        self.scales = scales
        self.degrees = self.parse_degrees(degrees)
        self.isotropic = isotropic
        self.interpolation = self.parse_interpolation(image_interpolation)

    def apply_transform(self, sample: dict) -> dict:
        check_consistent_shape(sample)
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
                interpolation = self.interpolation
            image_dict[DATA] = self.apply_affine_transform(
                image_dict[DATA],
                image_dict[AFFINE],
                scaling_params,
                rotation_params,
                interpolation,
            )
        return sample

    @staticmethod
    def get_params(
            scales: Tuple[float, float],
            degrees: Tuple[float, float],
            isotropic: bool,
            ) -> Tuple[List[float], List[float]]:
        scaling_params = torch.FloatTensor(3).uniform_(*scales)
        if isotropic:
            scaling_params.fill_(scaling_params[0])
        rotation_params = torch.FloatTensor(3).uniform_(*degrees)
        return scaling_params.tolist(), rotation_params.tolist()

    @staticmethod
    def get_scaling_transform(
            scaling_params: List[float],
            ) -> sitk.ScaleTransform:
        """
        scaling_params are inverted so that they are more intuitive
        For example, 1.5 means the objects look 1.5 times larger
        """
        transform = sitk.ScaleTransform(3)
        scaling_params = 1 / np.array(scaling_params)
        transform.SetScale(scaling_params)
        return transform

    @staticmethod
    def get_rotation_transform(
            degrees: List[float],
            ) -> sitk.Euler3DTransform:
        transform = sitk.Euler3DTransform()
        radians = np.radians(degrees)
        transform.SetRotation(*radians)
        return transform

    def apply_affine_transform(
            self,
            tensor: torch.Tensor,
            affine: np.ndarray,
            scaling_params: List[float],
            rotation_params: List[float],
            interpolation: Interpolation,
            ) -> torch.Tensor:
        assert tensor.ndim == 4
        assert len(tensor) == 1

        image = self.nib_to_sitk(tensor[0], affine)
        floating = reference = image

        scaling_transform = self.get_scaling_transform(scaling_params)
        rotation_transform = self.get_rotation_transform(rotation_params)
        transform = sitk.Transform(3, sitk.sitkComposite)
        transform.AddTransform(scaling_transform)
        transform.AddTransform(rotation_transform)

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolation.value)
        resampler.SetReferenceImage(reference)
        resampler.SetDefaultPixelValue(tensor.min().item())
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetTransform(transform)
        resampled = resampler.Execute(floating)

        np_array = sitk.GetArrayFromImage(resampled)
        np_array = np_array.transpose()  # ITK to NumPy
        tensor[0] = torch.from_numpy(np_array)
        return tensor
