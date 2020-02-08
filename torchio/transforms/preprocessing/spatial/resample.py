from typing import Tuple
import torch
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from ....utils import is_image_dict
from ....torchio import LABEL, DATA, AFFINE
from ... import Interpolation
from ... import Transform


class Resample(Transform):
    def __init__(
            self,
            voxel_sizes: Tuple[float, float, float],
            antialiasing: bool = True,
            image_interpolation: Interpolation = Interpolation.LINEAR,
            verbose: bool = False,
            ):
        super().__init__(verbose=verbose)
        self.voxel_sizes = voxel_sizes
        self.antialiasing = antialiasing
        self.image_interpolation = image_interpolation

    def apply_transform(self, sample: dict) -> dict:
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] == LABEL:
                interpolation = Interpolation.NEAREST
            else:
                interpolation = self.image_interpolation
            image_dict[DATA], image_dict[AFFINE] = self.apply_resample(
                image_dict[DATA],
                image_dict[AFFINE],
                self.voxel_sizes,
                interpolation,
            )
        return sample

    def apply_resample(
            self,
            tensor: torch.Tensor,
            affine: np.ndarray,
            voxel_sizes: Tuple[float, float, float],
            interpolation: Interpolation,
            ) -> Tuple[torch.Tensor, np.ndarray]:
        if interpolation == Interpolation.NEAREST:
            order = 0
        elif interpolation == Interpolation.LINEAR:
            order = 1
        elif interpolation == Interpolation.BSPLINE:
            order = 3
        else:
            message = f'Interpolation not implemented yet: {interpolation}'
            raise NotImplementedError(message)
        array = tensor.numpy()[0]
        nii = resample_to_output(
            nib.Nifti1Image(array, affine),
            voxel_sizes=voxel_sizes,
            order=order,
        )
        tensor = torch.from_numpy(nii.get_fdata(dtype=np.float32))
        tensor = tensor.unsqueeze(dim=0)
        return tensor, nii.affine
