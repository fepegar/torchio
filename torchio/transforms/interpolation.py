from enum import Enum
import SimpleITK as sitk


class Interpolation(Enum):
    """
    TODO: add more
    """
    NEAREST = sitk.sitkNearestNeighbor
    LINEAR = sitk.sitkLinear
    BSPLINE = sitk.sitkBSpline
