from itertools import cycle

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ..utils import to_tuple

class ImageSampler(IterableDataset):
    def __init__(self, sample, patch_size):
        """
        sample['image'] expected to have no batch dimensions
        """
        self.sample = sample
        self.patch_size = np.array(to_tuple(patch_size, n=3), dtype=np.uint16)

    def __iter__(self):
        return self.get_stream(self.sample, self.patch_size)

    def get_stream(self, sample, patch_size):
        """
        Is cycle neccesary?
        """
        return cycle(self.extract_patch_generator(sample, patch_size))

    def extract_patch_generator(self, sample, patch_size):
        while True:
            yield self.extract_patch(sample, patch_size)

    def extract_patch(self, sample, patch_size):
        index_ini, index_fin = self.get_random_indices(sample, patch_size)
        cropped_sample = self.copy_and_crop(
            sample,
            index_ini,
            index_fin,
        )
        return cropped_sample

    def get_random_indices(self, sample, patch_size):
        shape = np.array(sample['image'].shape[1:], dtype=np.uint16)
        max_index = shape - patch_size
        index = [
            torch.randint(i, size=(1,)).item() for i in max_index.tolist()
        ]
        index_ini = np.array(index, np.uint16)
        index_fin = index_ini + patch_size
        return index_ini, index_fin

    @staticmethod
    def crop(image, index_ini, index_fin):
        i_ini, j_ini, k_ini = index_ini
        i_fin, j_fin, k_fin = index_fin
        return image[..., i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]

    def copy_and_crop(self, sample, index_ini, index_fin):
        cropped_sample = {}
        for key, value in sample.items():
            if key in ('image', 'label'):
                cropped_sample[key] = self.crop(
                    value, index_ini, index_fin)
            else:
                cropped_sample[key] = value
        # torch doesn't like uint16
        cropped_sample['index_ini'] = index_ini.astype(int)
        return cropped_sample
