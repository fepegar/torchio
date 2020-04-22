import torch
import numpy as np
import nibabel as nib
from ....data.subject import Subject
from ....torchio import DATA, AFFINE
from ... import Transform


class ToCanonical(Transform):
    """Reorder the data to be closest to canonical (RAS+) orientation.

    This transform reorders the voxels and modifies the affine matrix so that
    the voxel orientations are nearest to:

        1. First voxel axis goes from left to Right
        2. Second voxel axis goes from posterior to Anterior
        3. Third voxel axis goes from inferior to Superior

    See `NiBabel docs about image orientation`_ for more information.

    Args:
        p: Probability that this transform will be applied.

    .. note:: The reorientation is performed using
        :py:meth:`nibabel.as_closest_canonical`.

    .. _NiBabel docs about image orientation: https://nipy.org/nibabel/image_orientation.html

    """

    def apply_transform(self, sample: Subject) -> dict:
        for image_dict in sample.get_images(intensity_only=False):
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
