from typing import Optional, List, Sequence, Union

import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from ..typing import TypeData
from ..data.subject import Subject
from ..data.image import Image, ScalarImage
from ..data.io import nib_to_sitk, sitk_to_nib


TypeTransformInput = Union[
    Subject,
    Image,
    torch.Tensor,
    np.ndarray,
    sitk.Image,
    dict,
    nib.Nifti1Image,
]


class DataParser:
    def __init__(
            self,
            data: TypeTransformInput,
            keys: Optional[Sequence[str]] = None,
            ):
        self.data = data
        self.keys = keys
        self.default_image_name = 'default_image_name'
        self.is_tensor = False
        self.is_array = False
        self.is_dict = False
        self.is_image = False
        self.is_sitk = False
        self.is_nib = False

    def get_subject(self):
        if isinstance(self.data, nib.Nifti1Image):
            tensor = self.data.get_fdata(dtype=np.float32)
            data = ScalarImage(tensor=tensor, affine=self.data.affine)
            subject = self._get_subject_from_image(data)
            self.is_nib = True
        elif isinstance(self.data, (np.ndarray, torch.Tensor)):
            subject = self._parse_tensor(self.data)
            self.is_array = isinstance(self.data, np.ndarray)
            self.is_tensor = True
        elif isinstance(self.data, Image):
            subject = self._get_subject_from_image(self.data)
            self.is_image = True
        elif isinstance(self.data, Subject):
            subject = self.data
        elif isinstance(self.data, sitk.Image):
            subject = self._get_subject_from_sitk_image(self.data)
            self.is_sitk = True
        elif isinstance(self.data, dict):  # e.g. Eisen or MONAI dicts
            if self.keys is None:
                message = (
                    'If the input is a dictionary, a value for "include" must'
                    ' be specified when instantiating the transform. See the'
                    ' docs for Transform:'
                    ' https://torchio.readthedocs.io/transforms/transforms.html#torchio.transforms.Transform'  # noqa: E501
                )
                raise RuntimeError(message)
            subject = self._get_subject_from_dict(self.data, self.keys)
            self.is_dict = True
        else:
            raise ValueError(f'Input type not recognized: {type(self.data)}')
        self._parse_subject(subject)
        return subject

    def get_output(self, transformed):
        if self.is_tensor or self.is_sitk:
            image = transformed[self.default_image_name]
            transformed = image.data
            if self.is_array:
                transformed = transformed.numpy()
            elif self.is_sitk:
                transformed = nib_to_sitk(image.data, image.affine)
        elif self.is_image:
            transformed = transformed[self.default_image_name]
        elif self.is_dict:
            transformed = dict(transformed)
            for key, value in transformed.items():
                if isinstance(value, Image):
                    transformed[key] = value.data
        elif self.is_nib:
            image = transformed[self.default_image_name]
            data = image.data
            if len(data) > 1:
                message = (
                    'Multichannel images not supported for input of type'
                    ' nibabel.nifti.Nifti1Image'
                )
                raise RuntimeError(message)
            transformed = nib.Nifti1Image(data[0].numpy(), image.affine)
        return transformed

    @staticmethod
    def _parse_subject(subject: Subject) -> None:
        if not isinstance(subject, Subject):
            message = (
                'Input to a transform must be a tensor or an instance'
                f' of torchio.Subject, not "{type(subject)}"'
            )
            raise RuntimeError(message)

    def _parse_tensor(self, data: TypeData) -> Subject:
        if data.ndim != 4:
            message = (
                'The input must be a 4D tensor with dimensions'
                f' (channels, x, y, z) but it has shape {tuple(data.shape)}'
            )
            raise ValueError(message)
        return self._get_subject_from_tensor(data)

    def _get_subject_from_tensor(self, tensor: torch.Tensor) -> Subject:
        image = ScalarImage(tensor=tensor)
        return self._get_subject_from_image(image)

    def _get_subject_from_image(self, image: Image) -> Subject:
        subject = Subject({self.default_image_name: image})
        return subject

    @staticmethod
    def _get_subject_from_dict(
            data: dict,
            image_keys: List[str],
            ) -> Subject:
        subject_dict = {}
        for key, value in data.items():
            if key in image_keys:
                value = ScalarImage(tensor=value)
            subject_dict[key] = value
        return Subject(subject_dict)

    def _get_subject_from_sitk_image(self, image):
        tensor, affine = sitk_to_nib(image)
        image = ScalarImage(tensor=tensor, affine=affine)
        return self._get_subject_from_image(image)
