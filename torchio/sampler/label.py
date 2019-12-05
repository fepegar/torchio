import copy
from itertools import cycle

import numpy as np
import torch
from torch.utils.data import IterableDataset


class LabelSampler(IterableDataset):
    def __init__(self, sample, patch_size):
        self.sample = sample
        self.patch_size = np.array(patch_size, dtype=np.uint16)

    def __iter__(self):
        return self.get_stream(self.sample, self.patch_size)

    def get_stream(self, sample, patch_size):
        return cycle(self.extract_patch(sample, patch_size))

    def extract_patch(self, sample, patch_size):
        while True:
            has_label = False
            while not has_label:
                index_ini = self.get_random_index(sample, patch_size)
                index_fin = index_ini + patch_size
                patch_label = self.crop(sample['label'], index_ini, index_fin)
                has_label = patch_label[1:].sum() > 0
            cropped_sample = self.copy_and_crop(
                sample,
                index_ini,
                index_fin,
            )
            yield cropped_sample

    def get_random_index(self, sample, patch_size):
        shape = np.array(sample['image'].shape[1:], dtype=np.uint16)
        max_index = shape - patch_size
        index = [
            torch.randint(i, size=(1,)).item() for i in max_index.tolist()
        ]
        return np.array(index, np.uint16)

    @staticmethod
    def crop(image, index_ini, index_fin):
        i_ini, j_ini, k_ini = index_ini
        i_fin, j_fin, k_fin = index_fin
        return image[..., i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]

    def copy_and_crop(self, sample, index_ini, index_fin):
        cropped_sample = copy.deepcopy(sample)
        cropped_sample['image'] = self.crop(
            cropped_sample['image'], index_ini, index_fin)
        cropped_sample['label'] = self.crop(
            cropped_sample['label'], index_ini, index_fin)
        return cropped_sample


# Example usage
def main():
    from itertools import islice
    from torch.utils.data import DataLoader

    sample = dict(
        image=torch.rand((1, 193, 229, 193)),
        label=torch.rand((2, 193, 229, 193)),
    )
    patch_size = 128, 128, 128
    samples_per_volume = 10

    sampler = LabelSampler(sample, patch_size)
    loader = DataLoader(sampler)

    for patch in islice(loader, samples_per_volume):
        print(patch)


if __name__ == "__main__":
    main()
