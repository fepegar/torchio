import enum
import SimpleITK as sitk


@enum.unique
class Interpolation(enum.IntEnum):
    NEAREST: int = sitk.sitkNearestNeighbor
    LINEAR: int = sitk.sitkLinear
    BSPLINE: int = sitk.sitkBSpline
    GAUSSIAN: int = sitk.sitkGaussian
    LABEL_GAUSSIAN: int = sitk.sitkLabelGaussian
    HAMMING: int = sitk.sitkHammingWindowedSinc
    COSINE: int = sitk.sitkCosineWindowedSinc
    WELCH: int = sitk.sitkWelchWindowedSinc
    LANCZOS: int = sitk.sitkLanczosWindowedSinc
    BLACKMAN: int = sitk.sitkBlackmanWindowedSinc
