from abc import ABC, abstractmethod

import torch


class RandomTransform(ABC):
    def __init__(self, seed=None, verbose=False):
        self.seed = seed
        self.verbose = verbose

    def __call__(self, sample):
        self.check_seed()
        if self.verbose:
            import time
            start = time.time()
        sample = self.apply_transform(sample)
        if self.verbose:
            duration = time.time() - start
            print(f'{self.__class__.__name__}: {duration:.1f} seconds')
        return sample

    @abstractmethod
    def apply_transform(self, sample):
        pass

    @staticmethod
    @abstractmethod
    def get_params():
        pass

    def check_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
