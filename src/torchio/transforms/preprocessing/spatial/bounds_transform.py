from ....transforms.transform import TypeBounds
from ...spatial_transform import SpatialTransform


class BoundsTransform(SpatialTransform):
    """Base class for transforms that change image bounds.

    Args:
        bounds_parameters: The meaning of this argument varies according to the
            child class.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, bounds_parameters: TypeBounds, **kwargs):
        super().__init__(**kwargs)
        self.bounds_parameters = self.parse_bounds(bounds_parameters)

    def is_invertible(self):
        return True
