import torch
from ...data.subject import Subject
from ...torchio import TypePatchSize
from .sampler import RandomSampler
from typing import Optional, Tuple, Generator
import numpy as np


class UniformSampler(RandomSampler):
    """Randomly extract patches from a volume with uniform probability.

    Args:
        patch_size: See :py:class:`~torchio.data.PatchSampler`.
    """
    def __init__(self, patch_size: TypePatchSize):
        super().__init__(patch_size)

    def get_probability_map(self, sample: Subject) -> torch.Tensor:
        return torch.ones(1, *sample.spatial_shape)

    def __call__(self, sample: Subject) -> Generator[Subject, None, None]:
        sample.check_consistent_spatial_shape()

        if np.any(self.patch_size > sample.spatial_shape):
            message = (
                f'Patch size {tuple(self.patch_size)} cannot be'
                f' larger than image size {tuple(sample.spatial_shape)}'
            )
            raise RuntimeError(message)

        valid_range = sample.spatial_shape - self.patch_size
        index_ini = [torch.randint(x + 1, (1,)).item() for x in valid_range]
        index_ini_array = np.asarray(index_ini)
        yield self.extract_patch(sample, index_ini_array)
