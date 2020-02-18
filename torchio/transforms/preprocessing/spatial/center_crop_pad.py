from typing import Union, Tuple, Optional, Dict
import numpy as np
import SimpleITK as sitk
from .pad import Pad
from .crop import Crop
from .bounds_transform import BoundsTransform
from ....torchio import DATA
from ....utils import is_image_dict, check_consistent_shape


class CenterCropOrPad(BoundsTransform):
    def __init__(
            self,
            size: Union[int, Tuple[int, int, int]],
            padding_mode: Optional[str] = None,
            padding_fill: Optional[float] = None,
            verbose: bool = False,
            ):
        super().__init__(size, verbose=verbose)
        self.padding_mode = padding_mode
        self.padding_fill = padding_fill

    def apply_transform(self, sample: dict) -> dict:
        source_shape = self.get_sample_shape(sample)
        target_shape = np.array(self.bounds_parameters[::2])  # hack
        diff_shape = target_shape - source_shape

        cropping = -np.minimum(diff_shape, 0)
        if cropping.any():
            cropping_params = self.get_six_bounds_parameters(cropping)
            sample = Crop(cropping_params)(sample)

        padding = np.maximum(diff_shape, 0)
        if padding.any():
            padding_kwargs: Dict[str, Optional[Union[str, float]]]
            padding_kwargs = {'fill': self.padding_fill}
            if self.padding_mode is not None:
                padding_kwargs['padding_mode'] = self.padding_mode
            padding_params = self.get_six_bounds_parameters(padding)
            sample = Pad(padding_params, **padding_kwargs)(sample)
        return sample

    @staticmethod
    def get_sample_shape(sample: dict) -> Tuple[int]:
        """Return the shape of the first image in the sample"""
        check_consistent_shape(sample)
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            data = image_dict[DATA].shape[1:]  # remove channels dimension
            break
        return data

    @staticmethod
    def get_six_bounds_parameters(parameters):
        parameters = parameters / 2
        result = []
        for n in parameters:
            ini, fin = int(np.ceil(n)), int(np.floor(n))
            result.extend([ini, fin])
        return tuple(result)
