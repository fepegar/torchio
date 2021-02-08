from numbers import Number
from typing import Union

import numpy as np
import nibabel as nib
import torch

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
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. _NumPy docs: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    """  # noqa: E501

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
            **kwargs
            ):
        super().__init__(padding, **kwargs)
        self.padding = padding
        self.check_padding_mode(padding_mode)
        self.padding_mode = padding_mode
        self.args_names = 'padding', 'padding_mode'

    @classmethod
    def check_padding_mode(cls, padding_mode):
        is_number = isinstance(padding_mode, Number)
        if not (padding_mode in cls.PADDING_MODES or is_number):
            message = (
                f'Padding mode "{padding_mode}" not valid. Valid options are'
                f' {list(cls.PADDING_MODES)} or a number'
            )
            raise KeyError(message)

    def apply_transform(self, subject: Subject) -> Subject:
        low = self.bounds_parameters[::2]
        for image in self.get_images(subject):
            new_origin = nib.affines.apply_affine(image.affine, -np.array(low))
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin
            if isinstance(self.padding_mode, Number):
                kwargs = {'mode': 'constant',
                          'constant_values': self.padding_mode}
            else:
                kwargs = {'mode': self.padding_mode}
            pad_params = self.bounds_parameters
            paddings = (0, 0), pad_params[:2], pad_params[2:4], pad_params[4:]
            padded = np.pad(image.data, paddings, **kwargs)
            image.set_data(torch.as_tensor(padded))
            image.affine = new_affine
        return subject

    def inverse(self):
        from .crop import Crop
        return Crop(self.padding)
