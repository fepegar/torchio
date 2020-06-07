from numbers import Number
from typing import Callable, Union
import SimpleITK as sitk
from .bounds_transform import BoundsTransform, TypeBounds


class Pad(BoundsTransform):
    r"""Pad an image.

    Args:
        padding: Tuple
            :math:`(d_{ini}, d_{fin}, h_{ini}, h_{fin}, w_{ini}, w_{fin})`
            defining the number of values padded to the edges of each axis.
            If the initial shape of the image is
            :math:`D \times H \times W`, the final shape will be
            :math:`(d_{ini} + D + d_{fin}) \times (h_{ini} + H + h_{fin}) \times (w_{ini} + W + w_{fin})`.
            If only three values :math:`(d, h, w)` are provided, then
            :math:`d_{ini} = d_{fin} = d`,
            :math:`h_{ini} = h_{fin} = h` and
            :math:`w_{ini} = w_{fin} = w`.
            If only one value :math:`n` is provided, then
            :math:`d_{ini} = d_{fin} = h_{ini} = h_{fin} = w_{ini} = w_{fin} = n`.
        padding_mode:
            Type of padding. Should be one of:

            - A number. Pad with a constant value.

            - ``reflect`` Pad with reflection of image without repeating the last value on the edge.

            - ``mirror`` Same as ``reflect``.

            - ``edge`` Pad with the last value at the edge of the image.

            - ``replicate`` Same as ``edge``.

            - ``circular`` Pad with the wrap of the vector along the axis. The first values are used to pad the end and the end values are used to pad the beginning.

            - ``wrap`` Same as ``circular``.

        p: Probability that this transform will be applied.

    """

    PADDING_FUNCTIONS = {
        'reflect': sitk.MirrorPad,
        'mirror': sitk.MirrorPad,
        'edge': sitk.ZeroFluxNeumannPad,
        'replicate': sitk.ZeroFluxNeumannPad,
        'circular': sitk.WrapPad,
        'wrap': sitk.WrapPad,
    }

    def __init__(
            self,
            padding: TypeBounds,
            padding_mode: Union[str, float] = 0,
            p: float = 1,
            ):
        """
        padding_mode can be 'constant', 'reflect', 'replicate' or 'circular'.
        See https://pytorch.org/docs/stable/nn.functional.html#pad for more
        information about this transform.
        """
        super().__init__(padding, p=p)
        self.padding_mode, self.fill = self.parse_padding_mode(padding_mode)

    @classmethod
    def parse_padding_mode(cls, padding_mode):
        if padding_mode in cls.PADDING_FUNCTIONS:
            fill = None
        elif isinstance(padding_mode, Number):
            fill = padding_mode
            padding_mode = 'constant'
        else:
            message = (
                f'Padding mode "{padding_mode}" not valid. Valid options are'
                f' {list(cls.PADDING_FUNCTIONS.keys())} or a number'
            )
            raise KeyError(message)
        return padding_mode, fill

    @property
    def bounds_function(self) -> Callable:
        if self.fill is not None:
            function = _pad_with_fill(self.fill)
        else:
            function = self.PADDING_FUNCTIONS[self.padding_mode]
        return function


def _pad_with_fill(fill):
    def wrapped(image, bounds1, bounds2):
        return sitk.ConstantPad(image, bounds1, bounds2, fill)
    return wrapped
