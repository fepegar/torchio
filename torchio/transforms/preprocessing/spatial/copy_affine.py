from ... import SpatialTransform
from ....data.image import LabelMap
from ....data.subject import Subject
import copy


class CopyAffine(SpatialTransform):
    """
    Copy the affine of a volume to all LabelMap.

    Args:
        target_key: copy the affine from this key name
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
            if isinstance(image, LabelMap):
                image.affine = copy.deepcopy(affine)
        return subject
