
import torch


class RandomNoise:
    def __init__(self, std_range=(0, 0.25), verbose=False):
        self.std_range = std_range
        self.verbose = verbose

    def __call__(self, sample):
        if self.verbose:
            import time
            start = time.time()
        std = self.get_params(self.std_range)
        sample['random_noise'] = std
        add_noise(sample['image'], std)
        if self.verbose:
            duration = time.time() - start
            print(f'RandomNoise: {duration:.1f} seconds')
        return sample

    @staticmethod
    def get_params(std_range):
        std = torch.FloatTensor(1).uniform_(*std_range).item()
        return std


def add_noise(data, std):
    noise = torch.FloatTensor(*data.shape).normal_(mean=0, std=std).numpy()
    data += noise
