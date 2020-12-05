from .queue import Queue
from .subject import Subject
from .dataset import SubjectsDataset
from .image import Image, ScalarImage, LabelMap
from .inference import GridSampler, GridAggregator
from .sampler import PatchSampler, LabelSampler, WeightedSampler, UniformSampler

Image.__module__ = "torchio.data"

__all__ = [
    'Queue',
    'Subject',
    'SubjectsDataset',
    'Image',
    'ScalarImage',
    'LabelMap',
    'GridSampler',
    'GridAggregator',
    'PatchSampler',
    'LabelSampler',
    'WeightedSampler',
    'UniformSampler',
]
