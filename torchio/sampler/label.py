"""
This iterable dataset yields patches that contain at least one voxel without
background. See main() for an example.

For now, this implementation is not efficient because it uses brute force to
look for foreground voxels.
"""

import torch

from .sampler import ImageSampler


class LabelSampler(ImageSampler):
    def extract_patch_generator(self, sample, patch_size):
        while True:
            yield self.extract_patch(sample, patch_size)

    def extract_patch(self, sample, patch_size):
        has_label = False
        while not has_label:
            index_ini, index_fin = self.get_random_indices(
                sample, patch_size)
            patch_label = self.crop(sample['label'], index_ini, index_fin)
            has_label = patch_label.sum() > 0
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
        if label.shape[1] > 1:
            raise NotImplementedError('Only labelmaps support is implemented')
