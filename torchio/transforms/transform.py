import time
import warnings
from abc import ABC, abstractmethod
import torch
import numpy as np
import SimpleITK as sitk
from ..utils import is_image_dict


FLIP_XY = np.diag((-1, -1, 1))


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
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        origin = np.dot(FLIP_XY, affine[:3, 3]).astype(np.float64)
        RZS = affine[:3, :3]
        spacing = np.sqrt(np.sum(RZS * RZS, axis=0))
        R = RZS / spacing
        direction = np.dot(FLIP_XY, R).flatten()
        image = sitk.GetImageFromArray(data.transpose())
        image.SetOrigin(origin)
        image.SetSpacing(spacing)
        image.SetDirection(direction)
        return image

    @staticmethod
    def sitk_to_nib(image):
        data = sitk.GetArrayFromImage(image).transpose()
        spacing = np.array(image.GetSpacing())
        R = np.array(image.GetDirection()).reshape(3, 3)
        R = np.dot(FLIP_XY, R)
        RZS = R * spacing
        translation = np.dot(FLIP_XY, image.GetOrigin())
        affine = np.eye(4)
        affine[:3, :3] = RZS
        affine[:3, 3] = translation
        return data, affine
