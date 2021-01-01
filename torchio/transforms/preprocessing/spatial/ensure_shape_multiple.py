from typing import Union, Optional

import numpy as np

from ... import SpatialTransform
from ....utils import to_tuple
from ....data.subject import Subject
from ....typing import TypeTripletInt
from .crop_or_pad import CropOrPad


class EnsureShapeMultiple(SpatialTransform):
    """Crop or pad an image to a shape that is a multiple of :math:`N`.

    Args:
        target_multiple: Tuple :math:`(w, h, d)`. If a single value :math:`n`
            is provided, then :math:`w = h = d = n`.
        method: Either ``'crop'`` or ``'pad'``.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> image = tio.datasets.Colin27().t1
        >>> image.shape
        (1, 181, 217, 181)
        >>> transform = tio.EnsureShapeMultiple(8, method='pad')
        >>> transformed = transform(image)
        >>> transformed.shape
        (1, 184, 224, 184)
        >>> transform = tio.EnsureShapeMultiple(8, method='crop')
        >>> transformed = transform(image)
        >>> transformed.shape
        (1, 176, 216, 176)
        >>> image_2d = image.data[..., :1]
        >>> image_2d.shape
        torch.Size([1, 181, 217, 1])
        >>> transformed = transform(image_2d)
        >>> transformed.shape
        torch.Size([1, 176, 216, 1])

    """
    def __init__(
            self,
            target_multiple: Union[int, TypeTripletInt],
            *,
            method: Optional[str] = 'pad',
            **kwargs
            ):
        super().__init__(**kwargs)
        self.target_multiple = np.array(to_tuple(target_multiple, 3))
        if method not in ('crop', 'pad'):
            raise ValueError('Method must be "crop" or "pad"')
        self.method = method

    def apply_transform(self, subject: Subject) -> Subject:
        source_shape = np.array(subject.spatial_shape, np.uint16)
        function = np.floor if self.method == 'crop' else np.ceil
        integer_ratio = function(source_shape / self.target_multiple)
        target_shape = integer_ratio * self.target_multiple
        target_shape = np.maximum(target_shape, 1)
        return CropOrPad(target_shape.astype(int))(subject)
