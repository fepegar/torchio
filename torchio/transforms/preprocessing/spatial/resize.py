import warnings

import numpy as np

from ....utils import to_tuple
from ....data.subject import Subject
from ....typing import TypeSpatialShape
from ... import SpatialTransform
from .resample import Resample
from .crop_or_pad import CropOrPad


class Resize(SpatialTransform):
    """Resample images so the output shape matches the given target shape.

    The field of view remains the same.

    Args:
        target_shape: Tuple :math:`(W, H, D)`. If a single value :math:`N` is
            provided, then :math:`W = H = D = N`.
        image_interpolation: See :ref:`Interpolation`.
    """
    def __init__(
            self,
            target_shape: TypeSpatialShape,
            image_interpolation: str = 'linear',
            **kwargs
            ):
        super().__init__(**kwargs)
        self.target_shape = np.asarray(to_tuple(target_shape, length=3))
        self.image_interpolation = self.parse_interpolation(
            image_interpolation)
        self.args_names = (
            'target_shape',
            'image_interpolation',
        )

    def apply_transform(self, subject: Subject) -> Subject:
        shape_in = np.asarray(subject.spatial_shape)
        shape_out = self.target_shape
        spacing_in = np.asarray(subject.spacing)
        spacing_out = shape_in / shape_out * spacing_in
        resample = Resample(
            spacing_out,
            image_interpolation=self.image_interpolation,
        )
        resampled = resample(subject)
        # Sometimes, the output shape is one voxel too large
        # Probably because Resample uses np.ceil to compute the shape
        if not resampled.spatial_shape == tuple(shape_out):
            message = (
                f'Output shape {resampled.spatial_shape}'
                f' != target shape {tuple(shape_out)}. Fixing with CropOrPad'
            )
            warnings.warn(message)
            crop_pad = CropOrPad(shape_out)
            resampled = crop_pad(resampled)
        return resampled
