import SimpleITK as sitk
from .bounds_transform import BoundsTransform


class Pad(BoundsTransform):
    PADDING_FUNCTIONS = {
        'constant': sitk.ConstantPad,
        'reflect': sitk.MirrorPad,
        'replicate': sitk.ZeroFluxNeumannPad,
        'circular': sitk.WrapPad,
        }

    def __init__(
            self,
            padding,
            padding_mode='constant',
            fill=None,
            verbose=False,
            ):
        """
        padding_mode can be 'constant', 'reflect', 'replicate' or 'circular'.
        See https://pytorch.org/docs/stable/nn.functional.html#pad for more
        information about this transform.
        """
        super().__init__(padding, verbose=verbose)
        self.padding_mode = padding_mode
        self.fill = fill

    @property
    def bounds_function(self):
        try:
            return self.PADDING_FUNCTIONS[self.padding_mode]
        except KeyError:
            message = (
                f'padding_mode "{self.padding_mode}" not valid.'
                f' Valid options are {list(self.PADDING_FUNCTIONS.keys())}'
            )
            raise ValueError(message)
