from typing import Union, Tuple, Optional, Dict
from .cropOrPad import CropOrPad


class CenterCropOrPad(CropOrPad):
    """Crop and/or pad an image to a target shape.
    Args:
        target_shape: Tuple :math:`(D, H, W)`. If a single value :math:`N` is
            provided, then :math:`D = H = W = N`.
        padding_mode: See :py:class:`~torchio.transforms.Pad`.
        padding_fill: Same as :attr:`fill` in
            :py:class:`~torchio.transforms.Pad`.
    """

    def __init__(
        self,
        target_shape: Union[int, Tuple[int, int, int]],
        padding_mode: str = 'constant',
        padding_fill: Optional[float] = None,
    ):
        super().__init__(target_shape=target_shape, padding_mode=padding_mode, padding_fill=padding_fill, mode="center")
