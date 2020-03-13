from numbers import Number
from typing import Union, Tuple
import torch
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
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
            It must be :py:attr:`torchio.Interpolation.NEAREST`,
            :py:attr:`torchio.Interpolation.LINEAR` or
            :py:attr:`torchio.Interpolation.BSPLINE`.

    .. note:: The resampling is performed using
        :py:meth:`nibabel.processing.resample_to_output`.

    """
    def __init__(
            self,
            target_spacing: TypeSpacing,
            antialiasing: bool = True,
            image_interpolation: Interpolation = Interpolation.LINEAR,
            ):
        super().__init__()
        self.target_spacing = self.parse_spacing(target_spacing)
        self.antialiasing = antialiasing
        self.interpolation_order = self.parse_interpolation(
            image_interpolation)

    @staticmethod
    def parse_spacing(spacing: TypeSpacing) -> Tuple[float, float, float]:
        if isinstance(spacing, tuple) and len(spacing) == 3:
            result = spacing
        elif isinstance(spacing, Number):
            result = 3 * (spacing,)
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

    def apply_transform(self, sample: dict) -> dict:
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

    def apply_resample(
            self,
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
