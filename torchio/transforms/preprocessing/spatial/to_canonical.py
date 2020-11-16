import torch
import numpy as np
import nibabel as nib
from ....data.subject import Subject
from ....torchio import DATA, AFFINE
from ... import SpatialTransform


class ToCanonical(SpatialTransform):
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
        :meth:`nibabel.as_closest_canonical`.

    .. _NiBabel docs about image orientation: https://nipy.org/nibabel/image_orientation.html
    """

    args_names = ()

    def apply_transform(self, subject: Subject) -> Subject:
        for image in subject.get_images(intensity_only=False):
            affine = image[AFFINE]
            if nib.aff2axcodes(affine) == tuple('RAS'):
                continue
            array = image[DATA].numpy()[np.newaxis]  # (1, C, W, H, D)
            # NIfTI images should have channels in 5th dimension
            array = array.transpose(2, 3, 4, 0, 1)  # (W, H, D, 1, C)
            nii = nib.Nifti1Image(array, affine)
            reoriented = nib.as_closest_canonical(nii)
            array = reoriented.get_fdata(dtype=np.float32)
            # https://github.com/facebookresearch/InferSent/issues/99#issuecomment-446175325
            array = array.copy()
            array = array.transpose(3, 4, 0, 1, 2)  # (1, C, W, H, D)
            image[DATA] = torch.from_numpy(array[0])
            image[AFFINE] = reoriented.affine
        return subject
