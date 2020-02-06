import time
import warnings
from abc import ABC, abstractmethod
import torch
import numpy as np
import SimpleITK as sitk
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

    @staticmethod
    def _fft_im(image):
        output = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
        return output

    @staticmethod
    def _ifft_im(freq_domain):
        output = np.fft.ifftshift(np.fft.ifftn(freq_domain))
        return output

    @staticmethod
    def _oversample(data, perc_oversampling=.10):
        """
        Oversamples data with a zero padding. Adds perc_oversampling percentage values
        :param data (ndarray): array to pad
        :param perc_oversampling (float): percentage of oversampling to add to data (based on its current shape)
        :return oversampled version of the data:
        """
        data_shape = list(data.shape)
        to_pad = np.ceil(np.asarray(data_shape) * perc_oversampling)
        left_pad = np.floor(to_pad / 2).astype(int)
        right_pad = np.ceil(to_pad / 2).astype(int)
        return np.pad(data, list(zip(left_pad, right_pad)))

    @staticmethod
    def crop_volume(data, cropping_shape):
        '''
        Cropping data to cropping_shape size. Cropping starts from center of the image
        '''
        vol_centers = (np.asarray(data.shape) / 2).astype(int)
        dim_ranges = np.ceil(np.asarray(cropping_shape) / 2).astype(int)
        slicing = [slice(dim_center - dim_range, dim_center + dim_range)
                   for dim_center, dim_range in zip(vol_centers, dim_ranges)]
        return data[slicing]
