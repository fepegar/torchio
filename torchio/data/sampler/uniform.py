import torch
from ...data.subject import Subject
from ...torchio import TypePatchSize
from .weighted import WeightedSampler


class UniformSampler(WeightedSampler):
    """Randomly extract patches from a volume with uniform probability.

    Args:
        patch_size: See :py:class:`~torchio.data.PatchSampler`.
    """
    def __init__(self, patch_size: TypePatchSize):
        super().__init__(patch_size)

    def get_probability_map(self, sample: Subject) -> torch.Tensor:
        return torch.ones(1, *sample.spatial_shape)
