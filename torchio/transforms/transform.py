import warnings
from copy import deepcopy
from abc import ABC, abstractmethod
import SimpleITK as sitk
from ..utils import is_image_dict, nib_to_sitk, sitk_to_nib
from .. import TypeData, TYPE, INTENSITY
import numpy as np

class Transform(ABC):
    """Abstract class for all TorchIO transforms.

    All classes used to transform a sample from an
    :py:class:`~torchio.ImagesDataset` should subclass it.
    All subclasses should overwrite
    :py:meth:`torchio.tranforms.Transform.apply_transform`,
    which takes a sample, applies some transformation and returns the result.
    """
    def __init__(self, verbose: bool = False, keep_original=False):
        self.verbose = verbose
        self.keep_original = keep_original

    def __call__(self, sample: dict):
        if self.keep_original:
            for image_name in list(sample):
                image_dict = sample[image_name]
                if not is_image_dict(image_dict):
                    continue
                if image_dict['type'] == INTENSITY:
                    new_key = image_name +'_orig'
                    if new_key not in sample:
                        sample[new_key] = dict(data=image_dict['data'], type='original', affine=image_dict['affine'])

        if self.verbose:
            start = time.time()
        """Transform a sample and return the result."""
        self.parse_sample(sample)
        sample = deepcopy(sample)
        sample = self.apply_transform(sample)
        return sample

    @abstractmethod
    def apply_transform(self, sample: dict):
        raise NotImplementedError

    @staticmethod
    def parse_sample(sample: dict) -> None:
        images_found = False
        type_in_dict = False
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            images_found = True
            if TYPE in image_dict:
                type_in_dict = True
            if images_found and type_in_dict:
                break
        if not images_found:
            warnings.warn(
                'No image dicts found in sample.'
                f' Sample keys: {sample.keys()}'
            )

    @staticmethod
    def nib_to_sitk(data: TypeData, affine: TypeData):
        return nib_to_sitk(data, affine)

    @staticmethod
    def sitk_to_nib(image: sitk.Image):
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
        to_pad = np.ceil(np.asarray(data_shape) * perc_oversampling/2) * 2
        #to force an even number if odd, this will shift the volume when croping
        #print("Pading at {}".format(to_pad))
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
        return data[tuple(slicing)]
