from typing import Union, Tuple
import torch
import numpy as np
from ....data.subject import Subject
from ....torchio import DATA, AFFINE
from ... import Transform


TypeShape = Tuple[int, int, int]
TypeSixBounds = Tuple[int, int, int, int, int, int]
TypeBounds = Union[
    int,
    TypeShape,
    TypeSixBounds,
]


class BoundsTransform(Transform):
    """Base class for transforms that change image bounds.

    Args:
        bounds_parameters: The meaning of this argument varies according to the
            child class.
        p: Probability that this transform will be applied.

    """
    def __init__(
            self,
            bounds_parameters: TypeBounds,
            p: float = 1,
            ):
        super().__init__(p=p)
        self.bounds_parameters = self.parse_bounds(bounds_parameters)

    @property
    def bounds_function(self):
        raise NotImplementedError

    @staticmethod
    def parse_bounds(bounds_parameters: TypeBounds) -> Tuple[int, ...]:
        try:
            bounds_parameters = tuple(bounds_parameters)
        except TypeError:
            bounds_parameters = (bounds_parameters,)

        # Check that numbers are integers
        for number in bounds_parameters:
            if not isinstance(number, (int, np.integer)) or number < 0:
                message = (
                    'Bounds values must be integers greater or equal to zero,'
                    f' not "{bounds_parameters}" of type {type(number)}'
                )
                raise ValueError(message)
        bounds_parameters = tuple(int(n) for n in bounds_parameters)
        bounds_parameters_length = len(bounds_parameters)
        if bounds_parameters_length == 6:
            return bounds_parameters
        if bounds_parameters_length == 1:
            return 6 * bounds_parameters
        if bounds_parameters_length == 3:
            return tuple(np.repeat(bounds_parameters, 2).tolist())
        message = (
            'Bounds parameter must be an integer or a tuple of'
            f' 3 or 6 integers, not {bounds_parameters}'
        )
        raise ValueError(message)

    def apply_transform(self, sample: Subject) -> dict:
        low = self.bounds_parameters[::2]
        high = self.bounds_parameters[1::2]
        for image_dict in sample.get_images(intensity_only=False):
            image = self.nib_to_sitk(image_dict[DATA][0], image_dict[AFFINE])
            result = self.bounds_function(image, low, high)
            data, affine = self.sitk_to_nib(result)
            tensor = torch.from_numpy(data).unsqueeze(0)
            image_dict[DATA] = tensor
            image_dict[AFFINE] = affine
        return sample
