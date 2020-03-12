import time
import warnings
from copy import deepcopy
from abc import ABC, abstractmethod
import SimpleITK as sitk
from ..utils import is_image_dict, nib_to_sitk, sitk_to_nib
from .. import TypeData, TYPE


class Transform(ABC):
    """Abstract class for all TorchIO transforms.

    All classes used to transform a sample from an
    :py:class:`~torchio.ImagesDataset` should subclass it.
    All subclasses should overwrite
    :py:meth:`torchio.tranforms.Transform.apply_transform`,
    which takes a sample, applies some transformation and returns the result.
    """
    def __call__(self, sample: dict):
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
