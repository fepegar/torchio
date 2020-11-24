from typing import Union, Tuple, Sequence, Optional
import numpy as np
from ....torchio import TypeTripletInt
from ... import SpatialTransform


TypeSixBounds = Tuple[int, int, int, int, int, int]
TypeBounds = Union[
    int,
    TypeTripletInt,
    TypeSixBounds,
]


class BoundsTransform(SpatialTransform):
    """Base class for transforms that change image bounds.

    Args:
        bounds_parameters: The meaning of this argument varies according to the
            child class.
        p: Probability that this transform will be applied.
        keys: See :class:`~torchio.transforms.Transform`.
    """
    def __init__(
            self,
            bounds_parameters: TypeBounds,
            p: float = 1,
            keys: Optional[Sequence[str]] = None,
            ):
        super().__init__(p=p, keys=keys)
        self.bounds_parameters = self.parse_bounds(bounds_parameters)

    def is_invertible(self):
        return True

    @staticmethod
    def parse_bounds(bounds_parameters: TypeBounds) -> TypeSixBounds:
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
