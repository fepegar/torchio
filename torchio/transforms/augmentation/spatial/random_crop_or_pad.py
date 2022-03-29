from typing import Union, Tuple, Optional

import numpy as np
from random import randint

from ...preprocessing.spatial.crop_or_pad import CropOrPad
from ...preprocessing.spatial.pad import Pad
from ...preprocessing.spatial.crop import Crop
from ... import SpatialTransform
from ...transform import TypeTripletInt, TypeSixBounds
from ....utils import parse_spatial_shape
from ....data.subject import Subject


class RandomCropOrPad(SpatialTransform):
    """Modify the field of view by random cropping or padding to a target shape.

    This transform modifies the affine matrix associated to the volume so that
    physical positions of the voxels are maintained.

    Args:
        target_shape: Tuple :math:`(W, H, D)`. If a single value :math:`N` is
            provided, then :math:`W = H = D = N`.
        padding_mode: Same as :attr:`padding_mode` in
            :class:`~torchio.transforms.Pad`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> subject = tio.Subject(
        ...     chest_ct=tio.ScalarImage('subject_a_ct.nii.gz'),
        ...     heart_mask=tio.LabelMap('subject_a_heart_seg.nii.gz'),
        ... )
        >>> subject.chest_ct.shape
        torch.Size([1, 512, 512, 289])
        >>> transform = tio.RandomCropOrPad(
        ...     (120, 80, 180)
        ... )
        >>> transformed = transform(subject)
        >>> transformed.chest_ct.shape
        torch.Size([1, 120, 80, 180])

    .. plot::

        import torchio as tio
        t1 = tio.datasets.Colin27().t1
        crop_pad = tio.RandomCropOrPad((256, 256, 32))
        t1_pad_crop = crop_pad(t1)
        subject = tio.Subject(t1=t1, crop_pad=t1_pad_crop)
        subject.plot()
    """  # noqa: E501

    def __init__(
        self,
        target_shape: Union[int, TypeTripletInt, None] = 16,
        padding_mode: Union[str, float] = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_shape = parse_spatial_shape(target_shape)
        self.padding_mode = padding_mode

    def _compute_random_cropping_padding_from_shapes(
        self, source_shape: TypeTripletInt,
    ) -> Tuple[Optional[TypeSixBounds], Optional[TypeSixBounds]]:
        diff_shape = np.array(self.target_shape) - source_shape

        cropping = -np.minimum(diff_shape, 0)
        if cropping.any():
            cropping_params = CropOrPad._get_six_bounds_parameters(cropping)
            # adjust the cropping params by a random amount
            # note: randint(0, 0) will return 0
            random_x = randint(-cropping[0] // 2, cropping[0] // 2)
            random_y = randint(-cropping[1] // 2, cropping[1] // 2)
            random_z = randint(-cropping[2] // 2, cropping[2] // 2)
            cropping_params = [
                cropping_params[0] + random_x,
                cropping_params[1] - random_x,
                cropping_params[2] + random_y,
                cropping_params[3] - random_y,
                cropping_params[4] + random_z,
                cropping_params[5] - random_z,
            ]
        else:
            cropping_params = None

        padding = np.maximum(diff_shape, 0)
        if padding.any():
            padding_params = CropOrPad._get_six_bounds_parameters(padding)
            # adjust the padding params by a random amount
            # note: randint(0, 0) will return 0
            random_x = randint(-padding[0] // 2, padding[0] // 2)
            random_y = randint(-padding[1] // 2, padding[1] // 2)
            random_z = randint(-padding[2] // 2, padding[2] // 2)
            padding_params = [
                padding_params[0] + random_x,
                padding_params[1] - random_x,
                padding_params[2] + random_y,
                padding_params[3] - random_y,
                padding_params[4] + random_z,
                padding_params[5] - random_z,
            ]
        else:
            padding_params = None

        return padding_params, cropping_params

    def apply_transform(self, subject: Subject) -> Subject:
        subject.check_consistent_space()
        source_shape = subject.spatial_shape
        (
            padding_params,
            cropping_params,
        ) = self._compute_random_cropping_padding_from_shapes(source_shape)
        padding_kwargs = {'padding_mode': self.padding_mode}

        if padding_params is not None:
            subject = Pad(padding_params, **padding_kwargs)(subject)
        if cropping_params is not None:
            subject = Crop(cropping_params)(subject)
        return subject
