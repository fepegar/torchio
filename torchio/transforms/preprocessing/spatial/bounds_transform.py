import torch
import numpy as np
from ....torchio import DATA, AFFINE
from ....utils import is_image_dict
from ... import Transform


class BoundsTransform(Transform):
    def __init__(
            self,
            bounds_parameters,
            verbose=False,
            ):
        """
        bounds_parameters should be an integer or a tuple of 3 or 6 integers
        """
        super().__init__(verbose=verbose)
        self.bounds_parameters = self.parse_bounds(bounds_parameters)

    @property
    def bounds_function(self):
        raise NotImplementedError

    def update_args(self, *args):
        return args

    def parse_bounds(self, bounds_parameters):
        try:
            bounds_parameters = tuple(bounds_parameters)
        except TypeError:
            bounds_parameters = (bounds_parameters,)
        bounds_parameters_length = len(bounds_parameters)
        if bounds_parameters_length == 6:
            return bounds_parameters
        elif bounds_parameters_length == 1:
            return 6 * bounds_parameters
        elif bounds_parameters_length == 3:
            return tuple(np.repeat(bounds_parameters, 2).tolist())
        message = (
            'Bounds parameter must be an integer or a tuple of'
            f' 3 or 6 integers, not {bounds_parameters}'
        )
        raise ValueError(message)

    def apply_transform(self, sample):
        low = self.bounds_parameters[::2]
        high = self.bounds_parameters[1::2]
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            image = self.nib_to_sitk(image_dict[DATA][0], image_dict[AFFINE])
            args = self.update_args(image, low, high)
            result = self.bounds_function(*args)
            data, affine = self.sitk_to_nib(result)
            tensor = torch.from_numpy(data).unsqueeze(0)
            image_dict[DATA] = tensor
            image_dict[AFFINE] = affine
        return sample
