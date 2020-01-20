import enum
import SimpleITK as sitk


@enum.unique
class Interpolation(enum.IntEnum):
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
