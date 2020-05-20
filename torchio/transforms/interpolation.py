import enum
import SimpleITK as sitk


@enum.unique
class Interpolation(enum.Enum):
    """Interpolation techniques available in ITK.

    Example:
        >>> from torchio.transforms import RandomAffine
        >>> transform = RandomAffine(image_interpolation='nearest')
    """
    #: Interpolates image intensity at a non-integer pixel position by copying the intensity for the nearest neighbor.
    NEAREST: str = 'sitkNearestNeighbor'

    #: Linearly interpolates image intensity at a non-integer pixel position.
    LINEAR: str = 'sitkLinear'

    #: Computes the B-spline interpolation weights over the support region of the B-spline.
    BSPLINE: str = 'sitkBSpline'

    GAUSSIAN: str = 'sitkGaussian'
    LABEL_GAUSSIAN: str = 'sitkLabelGaussian'

    HAMMING: str = 'sitkHammingWindowedSinc'
    COSINE: str = 'sitkCosineWindowedSinc'
    WELCH: str = 'sitkWelchWindowedSinc'
    LANCZOS: str = 'sitkLanczosWindowedSinc'
    BLACKMAN: str = 'sitkBlackmanWindowedSinc'


def get_sitk_interpolator(interpolation: Interpolation):
    return getattr(sitk, interpolation.value)
