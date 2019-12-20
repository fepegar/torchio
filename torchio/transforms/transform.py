import time
from abc import ABC, abstractmethod


class Transform(ABC):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, sample):
        if self.verbose:
            start = time.time()
        sample = self.apply_transform(sample)
        if self.verbose:
            duration = time.time() - start
            print(f'{self.__class__.__name__}: {duration:.3f} seconds')
        return sample

    @abstractmethod
    def apply_transform(self, sample):
        pass
