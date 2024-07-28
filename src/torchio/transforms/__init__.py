# noreorder

from .transform import Transform
from .fourier import FourierTransform
from .spatial_transform import SpatialTransform
from .intensity_transform import IntensityTransform
from .preprocessing.label.label_transform import LabelTransform

# Generic
from .lambda_transform import Lambda

# Augmentation
from .augmentation.composition import OneOf
from .augmentation.composition import Compose

from .augmentation.spatial import RandomFlip, Flip
from .augmentation.spatial import RandomAffine, Affine
from .augmentation.spatial import RandomAnisotropy
from .augmentation.spatial import RandomElasticDeformation, ElasticDeformation
from .augmentation.spatial import (
    RandomCombinedAffineElasticDeformation,
    CombinedAffineElasticDeformation,
)

from .augmentation.intensity import RandomSwap, Swap
from .augmentation.intensity import RandomBlur, Blur
from .augmentation.intensity import RandomNoise, Noise
from .augmentation.intensity import RandomSpike, Spike
from .augmentation.intensity import RandomGamma, Gamma
from .augmentation.intensity import RandomMotion, Motion
from .augmentation.intensity import RandomGhosting, Ghosting
from .augmentation.intensity import RandomBiasField, BiasField
from .augmentation.intensity import RandomLabelsToImage, LabelsToImage

# Preprocessing
from .preprocessing import Pad
from .preprocessing import Crop
from .preprocessing import Resize
from .preprocessing import Resample
from .preprocessing import CropOrPad
from .preprocessing import CopyAffine
from .preprocessing import ToCanonical
from .preprocessing import ZNormalization
from .preprocessing import RescaleIntensity
from .preprocessing import Clamp
from .preprocessing import Mask
from .preprocessing import EnsureShapeMultiple
from .preprocessing import HistogramStandardization
from .preprocessing.intensity.histogram_standardization import train_histogram
from .preprocessing import OneHot
from .preprocessing import Contour
from .preprocessing import RemapLabels
from .preprocessing import RemoveLabels
from .preprocessing import SequentialLabels
from .preprocessing import KeepLargestComponent


__all__ = [
    'Transform',
    'FourierTransform',
    'SpatialTransform',
    'IntensityTransform',
    'LabelTransform',
    'Lambda',
    'OneOf',
    'Compose',
    'RandomFlip',
    'Flip',
    'RandomAffine',
    'Affine',
    'RandomAnisotropy',
    'RandomElasticDeformation',
    'ElasticDeformation',
    'RandomCombinedAffineElasticDeformation',
    'CombinedAffineElasticDeformation',
    'RandomSwap',
    'Swap',
    'RandomBlur',
    'Blur',
    'RandomNoise',
    'Noise',
    'RandomSpike',
    'Spike',
    'RandomGamma',
    'Gamma',
    'RandomMotion',
    'Motion',
    'RandomGhosting',
    'Ghosting',
    'RandomBiasField',
    'BiasField',
    'RandomLabelsToImage',
    'LabelsToImage',
    'Pad',
    'Crop',
    'Resize',
    'Resample',
    'ToCanonical',
    'ZNormalization',
    'HistogramStandardization',
    'RescaleIntensity',
    'Clamp',
    'Mask',
    'CropOrPad',
    'CopyAffine',
    'EnsureShapeMultiple',
    'train_histogram',
    'OneHot',
    'Contour',
    'RemapLabels',
    'RemoveLabels',
    'SequentialLabels',
    'KeepLargestComponent',
]
