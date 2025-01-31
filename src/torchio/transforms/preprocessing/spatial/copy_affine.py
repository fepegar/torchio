import copy

from ....data.subject import Subject
from ...spatial_transform import SpatialTransform


class CopyAffine(SpatialTransform):
    """Copy the spatial metadata from a reference image in the subject.

    Small unexpected differences in spatial metadata across different images
    of a subject can arise due to rounding errors while converting formats.

    If the ``shape`` and ``orientation`` of the images are the same and their
    ``affine`` attributes are different but very similar, this transform can be
    used to avoid errors during safety checks in other transforms and samplers.

    Args:
        target: Name of the image within the subject whose affine matrix will
            be used.

    Example:
        >>> import torch
        >>> import torchio as tio
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> affine = np.diag((*(np.random.rand(3) + 0.5), 1))
        >>> t1 = tio.ScalarImage(tensor=torch.rand(1, 100, 100, 100), affine=affine)
        >>> # Let's simulate a loss of precision
        >>> # (caused for example by NIfTI storing spatial metadata in single precision)
        >>> bad_affine = affine.astype(np.float16)
        >>> t2 = tio.ScalarImage(tensor=torch.rand(1, 100, 100, 100), affine=bad_affine)
        >>> subject = tio.Subject(t1=t1, t2=t2)
        >>> resample = tio.Resample(0.5)
        >>> resample(subject).shape  # error as images are in different spaces
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "/Users/fernando/git/torchio/torchio/data/subject.py", line 101, in shape
            self.check_consistent_attribute('shape')
          File "/Users/fernando/git/torchio/torchio/data/subject.py", line 229, in check_consistent_attribute
            raise RuntimeError(message)
        RuntimeError: More than one shape found in subject images:
        {'t1': (1, 210, 244, 221), 't2': (1, 210, 243, 221)}
        >>> transform = tio.CopyAffine('t1')
        >>> fixed = transform(subject)
        >>> resample(fixed).shape
        (1, 210, 244, 221)


    .. warning:: This transform should be used with caution. Modifying the
        spatial metadata of an image manually can lead to incorrect processing
        of the position of anatomical structures. For example, a machine
        learning algorithm might incorrectly predict that a lesion on the right
        lung is on the left lung.

    .. note:: For more information, see some related discussions on GitHub:

        * https://github.com/TorchIO-project/torchio/issues/354
        * https://github.com/TorchIO-project/torchio/discussions/489
        * https://github.com/TorchIO-project/torchio/pull/584
        * https://github.com/TorchIO-project/torchio/issues/430
        * https://github.com/TorchIO-project/torchio/issues/382
        * https://github.com/TorchIO-project/torchio/pull/592
    """

    def __init__(self, target: str, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(target, str):
            message = f'The target must be a string, but "{type(target)}" was found'
            raise ValueError(message)
        self.target = target
        self.args_names = ['target']

    def apply_transform(self, subject: Subject) -> Subject:
        if self.target not in subject:
            message = f'Target image "{self.target}" not found in subject'
            raise RuntimeError(message)
        reference = subject[self.target]
        affine = copy.deepcopy(reference.affine)
        for image in self.get_images(subject):
            if image is reference:
                continue
            # We load the image to avoid complications
            # https://github.com/TorchIO-project/torchio/issues/1071#issuecomment-1511814720
            image.load()
            image.affine = affine
        return subject
