import enum

import SimpleITK as sitk


class Interpolation(enum.Enum):
    """Interpolation techniques available in ITK.

    For a full quantitative comparison of interpolation methods, you can read
    `Meijering et al. 1999, Quantitative Comparison of Sinc-Approximating Kernels for Medical Image Interpolation <https://link.springer.com/chapter/10.1007/10704282_23>`_

    Example:
        >>> import torchio as tio
        >>> transform = tio.RandomAffine(image_interpolation='bspline')
    """

    #: Interpolates image intensity at a non-integer pixel position by copying the intensity for the nearest neighbor.
    NEAREST = 'sitkNearestNeighbor'

    #: Linearly interpolates image intensity at a non-integer pixel position.
    LINEAR = 'sitkLinear'

    #: B-Spline of order 3 (cubic) interpolation.
    BSPLINE = 'sitkBSpline'

    #: Same as ``nearest``.
    CUBIC = 'sitkBSpline'

    #: Gaussian interpolation. Sigma is set to 0.8 input pixels and alpha is 4
    GAUSSIAN = 'sitkGaussian'

    #: Smoothly interpolate multi-label images. Sigma is set to 1 input pixel and alpha is 1
    LABEL_GAUSSIAN = 'sitkLabelGaussian'

    #: Hamming windowed sinc kernel.
    HAMMING = 'sitkHammingWindowedSinc'

    #: Cosine windowed sinc kernel.
    COSINE = 'sitkCosineWindowedSinc'

    #: Welch windowed sinc kernel.
    WELCH = 'sitkWelchWindowedSinc'

    #: Lanczos windowed sinc kernel.
    LANCZOS = 'sitkLanczosWindowedSinc'

    #: Blackman windowed sinc kernel.
    BLACKMAN = 'sitkBlackmanWindowedSinc'


def get_sitk_interpolator(interpolation: str) -> int:
    if not isinstance(interpolation, str):
        message = (
            f'Interpolation must be a string, not "{interpolation}"'
            f' of type {type(interpolation)}'
        )
        raise ValueError(message)
    string = getattr(Interpolation, interpolation.upper()).value
    return getattr(sitk, string)
