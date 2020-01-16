import time
import warnings
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from ..utils import is_image_dict


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
    def nib_to_sitk(array, affine):
        """
        TODO: figure out how to get directions
        from affine so that I don't need this
        """
        if isinstance(array, torch.Tensor):
            array = array.numpy()
        with NamedTemporaryFile(suffix='.nii') as f:
            nib.Nifti1Image(array, affine).to_filename(f.name)
            image = sitk.ReadImage(f.name)
        return image

    @staticmethod
    def sitk_to_nib(image):
        with NamedTemporaryFile(suffix='.nii') as f:
            sitk.WriteImage(image, f.name)
            nii = nib.load(f.name)
            data = nii.get_fdata(dtype=np.float32)
            affine = nii.affine
        return data, affine
