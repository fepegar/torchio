from typing import Union, Tuple, Optional, Dict
import numpy as np
import SimpleITK as sitk
from .pad import Pad
from .crop import Crop
from .bounds_transform import BoundsTransform
from ....torchio import DATA
from ....utils import is_image_dict, check_consistent_shape


class CenterCropOrPad(BoundsTransform):
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
        super().__init__(target_shape)
        self.padding_mode = Pad.parse_padding_mode(padding_mode)
        self.padding_fill = padding_fill

    def apply_transform(self, sample: dict) -> dict:
        source_shape = self._get_sample_shape(sample)
        # The parent class turns the 3-element shape tuple (d, h, w)
        # into a 6-element bounds tuple (d, d, h, h, w, w)
        target_shape = np.array(self.bounds_parameters[::2])
        diff_shape = target_shape - source_shape

        cropping = -np.minimum(diff_shape, 0)
        if cropping.any():
            cropping_params = self._get_six_bounds_parameters(cropping)
            sample = Crop(cropping_params)(sample)

        padding = np.maximum(diff_shape, 0)
        if padding.any():
            padding_kwargs: Dict[str, Optional[Union[str, float]]]
            padding_kwargs = {'fill': self.padding_fill}
            if self.padding_mode is not None:
                padding_kwargs['padding_mode'] = self.padding_mode
            padding_params = self._get_six_bounds_parameters(padding)
            sample = Pad(padding_params, **padding_kwargs)(sample)
        return sample

    @staticmethod
    def _get_sample_shape(sample: dict) -> Tuple[int]:
        """Return the shape of the first image in the sample."""
        check_consistent_shape(sample)
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            data = image_dict[DATA].shape[1:]  # remove channels dimension
            break
        return data

    @staticmethod
    def _get_six_bounds_parameters(parameters: np.ndarray):
        r"""Compute bounds parameters for ITK filters.

        Args:
            parameters: Tuple :math:`(d, h, w)` with the number of voxels to be
                cropped or padded.

        Returns:
            Tuple :math:`(d_{ini}, d_{fin}, h_{ini}, h_{fin}, w_{ini}, w_{fin})`,
            where :math:`n_{ini} = \left \lceil \frac{n}{2} \right \rceil` and
            :math:`n_{fin} = \left \lfloor \frac{n}{2} \right \rfloor`.

        Example:
            >>> p = np.array((4, 0, 7))
            >>> _get_six_bounds_parameters(p)
            (2, 2, 0, 0, 4, 3)

        """
        parameters = parameters / 2
        result = []
        for n in parameters:
            ini, fin = int(np.ceil(n)), int(np.floor(n))
            result.extend([ini, fin])
        return tuple(result)
