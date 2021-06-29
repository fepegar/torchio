from ... import SpatialTransform
from ....data.subject import Subject
import numpy as np
import copy


class CopyAffine(SpatialTransform):
    """
    Copy the affine of a target image to all images that have a
    different affine than the target.

    This transform should be used with caution: be sure that you know
    what you are doing.
    In general having different affines between the volumes
    need to be taken into account with resampling or cropping transform.
    For some special cases related to the rounding error in the affine nifti
    parameter or because the affine has been changed (for instance after a
    spm coregister) one may want to erase the original affine read from the
    nifti volume and take the one of another volume.
    But you must check if it is correct to copy the reference affine.
    This is fine if you have the same voxel grid and the same voxel ordering.

    Args:
        target_key: copy the affine from this image key name
    Example:
        >>> import torchio as tio
        >>> subject = tio.datasets.Colin27()
        >>> subject['t1'].affine *= 1.1
        >>> transform = tio.CopyAffine(target_key='t1')
        >>> transformed = transform(subject)
        >>> transformed['t1'].affine
        array([[   1.1,    0. ,    0. ,  -99. ],
               [   0. ,    1.1,    0. , -138.6],
               [   0. ,    0. ,    1.1,  -79.2],
               [   0. ,    0. ,    0. ,    1.1]])
        >>> transformed['head'].affine
        array([[   1.1,    0. ,    0. ,  -99. ],
               [   0. ,    1.1,    0. , -138.6],
               [   0. ,    0. ,    1.1,  -79.2],
               [   0. ,    0. ,    0. ,    1.1]])
        >>> transformed['brain'].affine
        array([[   1.1,    0. ,    0. ,  -99. ],
               [   0. ,    1.1,    0. , -138.6],
               [   0. ,    0. ,    1.1,  -79.2],
               [   0. ,    0. ,    0. ,    1.1]])
    """
    def __init__(self,
                 target_key: str,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_key = target_key
        self.args_names = ['target_key']

    def apply_transform(self, subject: Subject) -> Subject:
        affine = subject[self.target_key].affine
        for image in subject.get_images(intensity_only=False):
            if not np.array_equal(affine, image.affine):
                image.affine = copy.deepcopy(affine)
        return subject
