import torch
import numpy as np
import nibabel as nib
from ....utils import is_image_dict
from ....torchio import DATA, AFFINE
from ... import Transform


class ToCanonical(Transform):
    def __init__(
            self,
            verbose: bool = False,
            ):
        """Reorder the data to be closest to canonical (RAS+) orientation"""
        super().__init__(verbose=verbose)

    def apply_transform(self, sample: dict) -> dict:
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            affine = image_dict[AFFINE]
            if nib.aff2axcodes(affine) == tuple('RAS'):
                continue
            array = image_dict[DATA][0].numpy()
            nii = nib.Nifti1Image(array, affine)
            reoriented = nib.as_closest_canonical(nii)
            array = reoriented.get_fdata(dtype=np.float32)
            # https://github.com/facebookresearch/InferSent/issues/99#issuecomment-446175325
            array = array.copy()[np.newaxis, ...]
            image_dict[DATA] = torch.from_numpy(array)
            image_dict[AFFINE] = reoriented.affine
        return sample
