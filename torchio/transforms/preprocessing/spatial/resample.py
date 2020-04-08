from numbers import Number
from typing import Union, Tuple
import torch
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
from ....utils import is_image_dict
from ....torchio import LABEL, DATA, AFFINE, TYPE
from ... import Interpolation
from ... import Transform


TypeSpacing = Union[float, Tuple[float, float, float]]


class Resample(Transform):
    """Change voxel spacing keeping the field of view.

    Args:
        target_spacing: Tuple :math:`(s_d, s_h, s_w)`. If only one value
            :math:`n` is specified, then :math:`s_d = s_h = s_w = n`.
        antialiasing: (Not implemented yet).
        image_interpolation: Member of :py:class:`torchio.Interpolation`.
            Supported interpolation techniques for resampling are
            :py:attr:`torchio.Interpolation.NEAREST`,
            :py:attr:`torchio.Interpolation.LINEAR` and
            :py:attr:`torchio.Interpolation.BSPLINE`.
        reference_spacing: Str name of the image to which all voxelgrids will be resampled

    .. note:: The resampling is performed using
        :py:meth:`nibabel.processing.resample_to_output` and
        :py:meth:`nibabel.processing.resample_from_to`

    """
    def __init__(
            self,
            target_spacing: TypeSpacing,
            antialiasing: bool = True,
            image_interpolation: Interpolation = Interpolation.LINEAR,
            reference_image: str = None
            ):
        super().__init__()
        self.target_spacing = self.parse_spacing(target_spacing)
        self.antialiasing = antialiasing
        self.interpolation_order = self.parse_interpolation(
            image_interpolation)
        self.reference_image = reference_image

    @staticmethod
    def parse_spacing(spacing: TypeSpacing) -> Tuple[float, float, float]:
        if isinstance(spacing, tuple) and len(spacing) == 3:
            result = spacing
        elif isinstance(spacing, Number):
            result = 3 * (spacing,)
        elif spacing is None:
            result = None
        return result

    @staticmethod
    def parse_interpolation(interpolation: Interpolation) -> int:
        if interpolation == Interpolation.NEAREST:
            order = 0
        elif interpolation == Interpolation.LINEAR:
            order = 1
        elif interpolation == Interpolation.BSPLINE:
            order = 3
        else:
            message = f'Interpolation not implemented yet: {interpolation}'
            raise NotImplementedError(message)
        return order

    # @staticmethod
    # def parse_reference_image(sample: dict, reference_image: str) -> str:
    #     if reference_image is None:
    #         result = reference_image
    #     elif reference_image in sample.keys():
    #         result = reference_image
    #     else:
    #         message = f'reference_image={reference_image} not present in Subject, only these images were found: {sample.keys()} '
    #         raise ValueError(message)
    #     return result

    @staticmethod
    def check_reference_image(reference_image: str, sample: dict):
        if not isinstance(reference_image, str):
            message = f'reference_image argument should be of type str, type {type(reference_image)} was given'
            raise TypeError(message)
        if reference_image not in sample.keys():
            message = f'reference_image=\'{reference_image}\' not present in sample, only these keys were found: {sample.keys()}'
            raise ValueError(message)

    def apply_transform(self, sample: dict) -> dict:
        reference_image = self.reference_image
        if reference_image is not None:
            self.check_reference_image(reference_image, sample)
            reference_dict = sample[reference_image]
            # Only resample reference image if a target spacing is given
            if self.target_spacing is not None:
                if reference_dict[TYPE] == LABEL:
                    interpolation_order = 0
                else:
                    interpolation_order = self.interpolation_order
                reference_dict[DATA], reference_dict[AFFINE] = self.apply_resample(
                        reference_dict[DATA],
                        reference_dict[AFFINE],
                        self.target_spacing,
                        interpolation_order,
                    )

            for key, image_dict in sample.items():
                if not is_image_dict(image_dict):
                    continue
                if key in reference_image:
                    continue
                else:
                    image_dict[DATA], image_dict[AFFINE] = self.apply_reference_spacing(
                        image_dict[DATA],
                        image_dict[AFFINE],
                        reference_dict[DATA],
                        reference_dict[AFFINE],
                        interpolation_order,
                    )
        else:
            for image_dict in sample.values():
                if not is_image_dict(image_dict):
                    continue
                if image_dict[TYPE] == LABEL:
                    interpolation_order = 0  # nearest neighbor
                else:
                    interpolation_order = self.interpolation_order
                image_dict[DATA], image_dict[AFFINE] = self.apply_resample(
                    image_dict[DATA],
                    image_dict[AFFINE],
                    self.target_spacing,
                    interpolation_order,
                )
        return sample


    @staticmethod
    def apply_resample(
            tensor: torch.Tensor,
            affine: np.ndarray,
            target_spacing: Tuple[float, float, float],
            interpolation_order: int,
            ) -> Tuple[torch.Tensor, np.ndarray]:
        array = tensor.numpy()[0]
        nii = resample_to_output(
            nib.Nifti1Image(array, affine),
            voxel_sizes=target_spacing,
            order=interpolation_order,
        )
        tensor = torch.from_numpy(nii.get_fdata(dtype=np.float32))
        tensor = tensor.unsqueeze(dim=0)
        return tensor, nii.affine

    @staticmethod
    def apply_reference_spacing(
            tensor: torch.Tensor,
            affine: np.ndarray,
            reference_tensor: torch.Tensor,
            reference_affine: np.ndarray,
            interpolation_order: int,
            ) -> Tuple[torch.Tensor, np.ndarray]:
        array = tensor.numpy()[0]
        reference_array = reference_tensor.numpy()[0]
        nii = resample_from_to(
            nib.Nifti1Image(array, affine),
            nib.Nifti1Image(reference_array,reference_affine),
            order=interpolation_order,
        )
        tensor = torch.from_numpy(nii.get_fdata(dtype=np.float32))
        tensor = tensor.unsqueeze(dim=0)
        return tensor, nii.affine

