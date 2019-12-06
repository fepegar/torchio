"""
This iterable dataset yields patches that contain at least one voxel without
background. See main() for an example.

For now, this implementation is not efficient because it uses brute force to
look for foreground voxels.
"""

import copy
from typing import Union
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .sampler import ImageSampler


class LabelSampler(ImageSampler):
    def __init__(self, sample, patch_size):
        super().__init__(sample, patch_size)

    def extract_patch_generator(self, sample, patch_size):
        while True:
            yield self.extract_patch(sample, patch_size)

    def extract_patch(self, sample, patch_size):
        has_label = False
        while not has_label:
            index_ini, index_fin = self.get_random_indices(
                sample, patch_size)
            patch_label = self.crop(sample['label'], index_ini, index_fin)
            foreground = patch_label[1:]
            has_label = foreground.sum() > 0
        cropped_sample = self.copy_and_crop(
            sample,
            index_ini,
            index_fin,
        )
        return cropped_sample

    @staticmethod
    def get_data_bounds(label: torch.Tensor):
        if label.dims() != 4:
            raise NotImplementedError('Only 3D images is implemented')
        if label.shape[1] > 2:
            raise NotImplementedError('Only one foreground class is implemented')


# # Example usage
# import torch
# from torchio.sampler import LabelSampler, ImageSampler
# from itertools import islice
# from torch.utils.data import DataLoader

# shape = 193, 229, 193
# foreground = torch.zeros(*shape)
# foreground[50, 50, 50] = 1
# background = 1 - foreground
# label = torch.stack((background, foreground))

# sample = dict(
#     image=torch.rand((1, 193, 229, 193)),
#     # label=torch.rand((2, 193, 229, 193)),
#     label=label,
# )
# patch_size = 128, 128, 128
# samples_per_volume = 10

# class_ = ImageSampler
# sampler = class_(sample, patch_size)
# loader = DataLoader(sampler)

# for patch_sample in islice(loader, samples_per_volume):
#     print(patch_sample['index_ini'])
