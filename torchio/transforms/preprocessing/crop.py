import SimpleITK as sitk
from .bounds_transform import BoundsTransform


class Crop(BoundsTransform):
    def __init__(
            self,
            cropping,
            verbose=False,
            ):
        """
        cropping should be an integer or a tuple of 3 or 6 integers
        """
        super().__init__(cropping, verbose=verbose)

    @property
    def bounds_function(self):
        return sitk.Crop
