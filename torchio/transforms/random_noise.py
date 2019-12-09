
import torch
from .random_transform import RandomTransform


class RandomNoise(RandomTransform):
    def __init__(self, std_range=(0, 0.25), seed=None, verbose=False):
        super().__init__(seed=seed, verbose=verbose)
        self.std_range = std_range
        self.seed = seed
        self.verbose = verbose

    def apply_transform(self, sample):
        std = self.get_params(self.std_range)
        sample['random_noise'] = std
        add_noise(sample['image'], std)
        return sample

    @staticmethod
    def get_params(std_range):
        std = torch.FloatTensor(1).uniform_(*std_range).item()
        return std


def add_noise(data, std):
    noise = torch.FloatTensor(*data.shape).normal_(mean=0, std=std).numpy()
    data += noise
