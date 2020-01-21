import copy
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ...torchio import DATA
from ...utils import to_tuple, is_image_dict, check_consistent_shape


class ImageSampler(IterableDataset):
    def __init__(self, sample, patch_size):
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
        check_consistent_shape(sample)
        first_image_name = list(sample.keys())[0]
        first_image_array = sample[first_image_name][DATA]
        # first_image_array should have shape (1, H, W, D)
        shape = np.array(first_image_array.shape[1:], dtype=np.uint16)
        max_index_ini = shape - patch_size
        if (max_index_ini < 0).any():
            message = (
                f'Patch size {patch_size} must not be'
                f' larger than image size {tuple(shape)}'
            )
            raise ValueError(message)
        coordinates = []
        for max_coordinate in max_index_ini.tolist():
            if max_coordinate == 0:
                coordinate = 0
            else:
                coordinate = torch.randint(max_coordinate, size=(1,)).item()
            coordinates.append(coordinate)
        index_ini = np.array(coordinates, np.uint16)
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
            cropped_sample[key] = copy.copy(value)
            if is_image_dict(value):
                sample_image_dict = value
                cropped_image_dict = cropped_sample[key]
                cropped_image_dict[DATA] = self.crop(
                    sample_image_dict[DATA], index_ini, index_fin)
        # torch doesn't like uint16
        cropped_sample['index_ini'] = index_ini.astype(int)
        return cropped_sample
