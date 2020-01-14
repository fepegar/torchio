import numpy as np
from ...torchio import DATA
from ...utils import is_image_dict
from .. import Transform


class Crop(Transform):
    def __init__(
            self,
            cropping,
            verbose=False,
            ):
        """
        Cropping should be an integer or a tuple of 3 or 6 integers
        """
        super().__init__(verbose=verbose)
        self.cropping = self.parse_cropping(cropping)

    def parse_cropping(self, cropping):
        try:
            cropping = tuple(cropping)
        except TypeError:
            cropping = (cropping,)
        cropping_length = len(cropping)
        if cropping_length == 6:
            return cropping
        elif cropping_length == 1:
            return 6 * cropping
        elif cropping_length == 3:
            return tuple(np.repeat(cropping, 2))
        message = (
            '"cropping" must be an integer or a tuple of 3 or 6 integers'
            f' not {cropping}'
        )
        raise ValueError(message)

    def apply_transform(self, sample):
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            a, b, c, d, e, f = self.cropping
            image_dict[DATA] = image_dict[DATA][:, a:-b, c:-d, e:-f]
        return sample
