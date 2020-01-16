
import torch
from ....utils import is_image_dict
from ....torchio import DATA, INTENSITY
from .. import RandomTransform


class RandomNoise(RandomTransform):
    def __init__(self, std_range=(0, 0.25), seed=None, verbose=False):
        super().__init__(seed=seed, verbose=verbose)
        self.std_range = std_range

    def apply_transform(self, sample):
        std = self.get_params(self.std_range)
        sample['random_noise'] = std
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] != INTENSITY:
                continue
            image_dict[DATA] = add_noise(image_dict[DATA], std)
        return sample

    @staticmethod
    def get_params(std_range):
        std = torch.FloatTensor(1).uniform_(*std_range).item()
        return std


def add_noise(data, std):
    noise = torch.FloatTensor(*data.shape).normal_(mean=0, std=std)
    data = data + noise
    return data
