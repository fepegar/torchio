import torch
from ...data.subject import Subject
from ...typing import TypePatchSize
from .sampler import RandomSampler
from typing import Generator
import numpy as np


class UniformSampler(RandomSampler):
    """Randomly extract patches from a volume with uniform probability.

    Args:
        patch_size: See :class:`~torchio.data.PatchSampler`.
    """
    def __init__(self, patch_size: TypePatchSize):
        super().__init__(patch_size)

    def get_probability_map(self, subject: Subject) -> torch.Tensor:
        return torch.ones(1, *subject.spatial_shape)

    def __call__(
            self,
            subject: Subject,
            num_patches: int = None,
            ) -> Generator[Subject, None, None]:
        subject.check_consistent_spatial_shape()

        if np.any(self.patch_size > subject.spatial_shape):
            message = (
                f'Patch size {tuple(self.patch_size)} cannot be'
                f' larger than image size {tuple(subject.spatial_shape)}'
            )
            raise RuntimeError(message)

        valid_range = subject.spatial_shape - self.patch_size
        patches_left = num_patches if num_patches is not None else True
        while patches_left:
            index_ini = [
                torch.randint(x + 1, (1,)).item()
                for x in valid_range
            ]
            index_ini_array = np.asarray(index_ini)
            yield self.extract_patch(subject, index_ini_array)
            if num_patches is not None:
                patches_left -= 1
