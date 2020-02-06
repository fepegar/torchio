import time
import warnings
from abc import ABC, abstractmethod
from ..utils import is_image_dict, nib_to_sitk, sitk_to_nib


class Transform(ABC):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, sample):
        if self.verbose:
            start = time.time()
        self.parse_sample(sample)
        sample = self.apply_transform(sample)
        if self.verbose:
            duration = time.time() - start
            print(f'{self.__class__.__name__}: {duration:.3f} seconds')
        return sample

    @abstractmethod
    def apply_transform(self, sample):
        pass

    @staticmethod
    def parse_sample(sample):
        images_found = False
        type_in_dict = False
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            images_found = True
            if 'type' in image_dict:
                type_in_dict = True
            if images_found and type_in_dict:
                break
        if not images_found:
            warnings.warn(
                'No image dicts found in sample.'
                f' Sample keys: {sample.keys()}'
            )

    @staticmethod
    def nib_to_sitk(data, affine):
        return nib_to_sitk(data, affine)

    @staticmethod
    def sitk_to_nib(image):
        return sitk_to_nib(image)
