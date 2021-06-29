from ... import SpatialTransform
from ....data.subject import Subject
import numpy as np
import copy


class CopyAffine(SpatialTransform):
    """
    Copy the affine of a target image to all images that have a
    different affine than the target.

    Args:
        target_key: copy the affine from this image key name
    Example:
    >>> import torchio as tio
    >>> subject = tio.datasets.Colin27()
    >>> transform = tio.CopyAffine(target_key='t1')
    >>> transformed = transform(subject)
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
