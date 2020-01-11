from enum import Enum
import SimpleITK as sitk


class Interpolation(Enum):
    NEAREST = sitk.sitkNearestNeighbor
    LINEAR = sitk.sitkLinear
    BSPLINE = sitk.sitkBSpline
    GAUSSIAN = sitk.sitkGaussian
    LABEL_GAUSSIAN = sitk.sitkLabelGaussian
    HAMMING = sitk.sitkHammingWindowedSinc
    COSINE = sitk.sitkCosineWindowedSinc
    WELCH = sitk.sitkWelchWindowedSinc
    LANCZOS = sitk.sitkLanczosWindowedSinc
    BLACKMAN = sitk.sitkBlackmanWindowedSinc
