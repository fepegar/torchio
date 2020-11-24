from numbers import Number
from typing import Union, Sequence, Optional

import numpy as np
import nibabel as nib
import torch

from ....torchio import DATA, AFFINE
from ....data.subject import Subject
from .bounds_transform import BoundsTransform, TypeBounds


class Pad(BoundsTransform):
    r"""Pad an image.

    Args:
        padding: Tuple
            :math:`(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})`
            defining the number of values padded to the edges of each axis.
            If the initial shape of the image is
            :math:`W \times H \times D`, the final shape will be
            :math:`(w_{ini} + W + w_{fin}) \times (h_{ini} + H + h_{fin})
            \times (d_{ini} + D + d_{fin})`.
            If only three values :math:`(w, h, d)` are provided, then
            :math:`w_{ini} = w_{fin} = w`,
            :math:`h_{ini} = h_{fin} = h` and
            :math:`d_{ini} = d_{fin} = d`.
            If only one value :math:`n` is provided, then
            :math:`w_{ini} = w_{fin} = h_{ini} = h_{fin} =
            d_{ini} = d_{fin} = n`.
        padding_mode: See possible modes in `NumPy docs`_. If it is a number,
            the mode will be set to ``'constant'``.
        p: Probability that this transform will be applied.
        keys: See :class:`~torchio.transforms.Transform`.

    .. _NumPy docs: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    """

    PADDING_MODES = (
        'empty',
        'edge',
        'wrap',
        'constant',
        'linear_ramp',
        'maximum',
        'mean',
        'median',
        'minimum',
        'reflect',
        'symmetric',
    )

    def __init__(
            self,
            padding: TypeBounds,
            padding_mode: Union[str, float] = 0,
            p: float = 1,
            keys: Optional[Sequence[str]] = None,
            ):
        super().__init__(padding, p=p, keys=keys)
        self.padding = padding
        self.padding_mode, self.fill = self.parse_padding_mode(padding_mode)
        self.args_names = 'padding', 'padding_mode'

    @classmethod
    def parse_padding_mode(cls, padding_mode):
        if padding_mode in cls.PADDING_MODES:
            fill = None
        elif isinstance(padding_mode, Number):
            fill = padding_mode
            padding_mode = 'constant'
        else:
            message = (
                f'Padding mode "{padding_mode}" not valid. Valid options are'
                f' {list(cls.PADDING_MODES)} or a number'
            )
            raise KeyError(message)
        return padding_mode, fill

    def apply_transform(self, subject: Subject) -> Subject:
        low = self.bounds_parameters[::2]
        for image in self.get_images(subject):
            new_origin = nib.affines.apply_affine(image.affine, -np.array(low))
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin
            kwargs = {'mode': self.padding_mode}
            if self.padding_mode == 'constant':
                kwargs['constant_values'] = self.fill
            pad_params = self.bounds_parameters
            paddings = (0, 0), pad_params[:2], pad_params[2:4], pad_params[4:]
            padded = np.pad(image[DATA], paddings, **kwargs)
            image[DATA] = torch.from_numpy(padded)
            image[AFFINE] = new_affine
        return subject

    def inverse(self):
        from .crop import Crop
        return Crop(self.padding)
