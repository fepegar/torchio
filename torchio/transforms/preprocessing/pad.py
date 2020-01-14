import torch.nn.functional as F
import numpy as np
from ...torchio import DATA
from ...utils import is_image_dict
from .. import Transform


class Pad(Transform):
    def __init__(
            self,
            padding,
            fill=None,
            padding_mode=None,
            verbose=False,
            ):
        """
        padding_mode can be 'constant', 'reflect', 'replicate' or 'circular'.
        See https://pytorch.org/docs/stable/nn.functional.html#pad for more
        information about this transform.
        """
        super().__init__(verbose=verbose)
        self.padding_mode = padding_mode
        self.padding = self.parse_padding(padding)
        self.fill = fill

    def parse_padding(self, padding):
        """
        We need to check the padding mode because of this line:
        https://github.com/pytorch/pytorch/blob/b0ac425dc4a1340b82b1c58391fb1e4718815617/torch/nn/functional.py#L2923
        """
        try:
            padding = tuple(padding)
        except TypeError:
            padding = (padding,)
        is_constant = self.padding_mode is None
        padding_length = len(padding)
        if padding_length == 6:  # TODO: what if 6 and not constant?
            return padding
        elif padding_length == 1:
            if is_constant:
                return 6 * padding
            else:
                return 4 * padding
        elif padding_length == 3:
            if is_constant:
                return tuple(np.repeat(padding, 2))
            else:
                return (0,) + padding
        message = (
            '"padding" must be an integer or a tuple of 3 or 6 integers'
            f' not {padding}'
        )
        raise ValueError(message)

    def apply_transform(self, sample):
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            kwargs = {}
            if self.padding_mode is not None:
                kwargs['mode'] = self.padding_mode
            if self.fill is not None:
                kwargs['value'] = self.fill
            image_dict[DATA] = F.pad(
                image_dict[DATA],
                self.padding,
                **kwargs,
            )
        return sample
