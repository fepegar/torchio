import copy
import numbers
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List

import torch
import numpy as np
import SimpleITK as sitk

from .. import TypeData, DATA, TypeNumber
from ..data.subject import Subject
from ..data.image import Image, ScalarImage
from ..data.dataset import ImagesDataset
from ..utils import nib_to_sitk, sitk_to_nib
from .interpolation import Interpolation


IMAGE_NAME = 'image'


class Transform(ABC):
    image_name = 'image'
    """Abstract class for all TorchIO transforms.

    All classes used to transform a sample from an
    :py:class:`~torchio.ImagesDataset` should subclass it.
    All subclasses should overwrite
    :py:meth:`torchio.tranforms.Transform.apply_transform`,
    which takes a sample, applies some transformation and returns the result.

    Args:
        p: Probability that this transform will be applied.
        copy: Make a deep copy of the input before applying the transform.
        keys: If the input is a dictionary, the corresponding values will be
            converted to :py:class:`torchio.ScalarImage` so that the transform
            is applied to them only.
    """
    def __init__(
            self,
            p: float = 1,
            copy: bool = True,
            keys: Optional[List[str]] = None,
            ):
        self.probability = self.parse_probability(p)
        self.copy = copy
        self.keys = keys

    def __call__(self, data: Union[Subject, torch.Tensor, np.ndarray]):
        """Transform a sample and return the result.

        Args:
            data: Instance of :py:class:`~torchio.Subject`, 4D
                :py:class:`torch.Tensor` or 4D NumPy array with dimensions
                :math:`(C, D, H, W)`, where :math:`C` is the number of channels
                and :math:`D, H, W` are the spatial dimensions. If the input is
                a tensor, the affine matrix is an identity and a tensor will be
                also returned.
        """
        if torch.rand(1).item() > self.probability:
            return data

        is_dict = False
        if isinstance(data, (np.ndarray, torch.Tensor)):
            is_array = isinstance(data, np.ndarray)
            is_tensor = True
            is_image = False
            sample = self.parse_tensor(data)
        elif isinstance(data, Image):
            sample = self._get_subject_from_image(data)
            is_tensor = is_array = False
            is_image = True
        elif isinstance(data, Subject):
            is_tensor = is_array = is_image = False
            sample = data
        elif isinstance(data, dict):  # e.g. Eisen dict
            if self.keys is None:
                message = (
                    'If input is a dictionary, a value for "keys" must be'
                    ' specified when instantiating the transform'
                )
                raise RuntimeError(message)
            sample = self._get_subject_from_dict(data, self.keys)
            is_tensor = is_array = is_image = False
            is_dict = True
        self.parse_sample(sample)

        if self.copy:
            sample = copy.copy(sample)

        with np.errstate(all='raise'):
            transformed = self.apply_transform(sample)

        for image in transformed.get_images(intensity_only=False):
            ndim = image[DATA].ndim
            assert ndim == 4, f'Output of {self.name} is {ndim}D'

        if is_tensor:
            transformed = transformed[self.image_name][DATA]
        if is_array:
            transformed = transformed.numpy()
        if is_image:
            transformed = transformed[IMAGE_NAME]
        if is_dict:
            transformed = dict(transformed)
            for key, value in transformed.items():
                if isinstance(value, Image):
                    transformed[key] = value.data
        return transformed

    @abstractmethod
    def apply_transform(self, sample: Subject):
        raise NotImplementedError

    @staticmethod
    def parse_range(
            nums_range: Union[TypeNumber, Tuple[TypeNumber, TypeNumber]],
            name: str,
            min_constraint: TypeNumber = None,
            max_constraint: TypeNumber = None,
            type_constraint: type = None,
            ) -> Tuple[TypeNumber, TypeNumber]:
        r"""Adapted from ``torchvision.transforms.RandomRotation``.

        Args:
            nums_range: Tuple of two numbers :math:`(n_{min}, n_{max})`,
                where :math:`n_{min} \leq n_{max}`.
                If a single positive number :math:`n` is provided,
                :math:`n_{min} = -n` and :math:`n_{max} = n`.
            name: Name of the parameter, so that an informative error message
                can be printed.
            min_constraint: Minimal value that :math:`n_{min}` can take,
                default is None, i.e. there is no minimal value.
            max_constraint: Maximal value that :math:`n_{max}` can take,
                default is None, i.e. there is no maximal value.
            type_constraint: Precise type that :math:`n_{max}` and
                :math:`n_{min}` must take.

        Returns:
            A tuple of two numbers :math:`(n_{min}, n_{max})`.

        Raises:
            ValueError: if :attr:`nums_range` is negative
            ValueError: if :math:`n_{max}` or :math:`n_{min}` is not a number
            ValueError: if :math:`n_{max} \lt n_{min}`
            ValueError: if :attr:`min_constraint` is not None and
                :math:`n_{min}` is smaller than :attr:`min_constraint`
            ValueError: if :attr:`max_constraint` is not None and
                :math:`n_{max}` is greater than :attr:`max_constraint`
            ValueError: if :attr:`type_constraint` is not None and
                :math:`n_{max}` and :math:`n_{max}` are not of type
                :attr:`type_constraint`.
        """
        if isinstance(nums_range, numbers.Number):
            if nums_range < 0:
                raise ValueError(
                    f'If {name} is a single number,'
                    f' it must be positive, not {nums_range}')
            if min_constraint is not None and nums_range < min_constraint:
                raise ValueError(
                    f'If {name} is a single number, it must be greater'
                    f'than {min_constraint}, not {nums_range}'
                )
            if max_constraint is not None and nums_range > max_constraint:
                raise ValueError(
                    f'If {name} is a single number, it must be smaller'
                    f'than {max_constraint}, not {nums_range}'
                )
            if type_constraint is not None and \
                    not isinstance(nums_range, type_constraint):
                raise ValueError(
                    f'If {name} is a single number, it must be of'
                    f'type {type_constraint}, not {nums_range}'
                )
            min_range = -nums_range if min_constraint is None else nums_range
            return (min_range, nums_range)

        try:
            min_degree, max_degree = nums_range
        except (TypeError, ValueError):
            raise ValueError(
                f'If {name} is not a single number, it muste be'
                f'a sequence of len 2, not {nums_range}'
            )

        if not isinstance(min_degree, numbers.Number) or \
                not isinstance(max_degree, numbers.Number):
            message = (
                f'{name} values must be numbers, not {nums_range}')
            raise ValueError(message)

        if min_degree > max_degree:
            raise ValueError(
                f'If {name} is a sequence, the second value must be'
                f' equal or greater than the first, not {nums_range}')

        if min_constraint is not None and min_degree < min_constraint:
            raise ValueError(
                f'If {name} is a sequence, the first value must be greater'
                f'than {min_constraint}, not {min_degree}'
            )

        if max_constraint is not None and max_degree > max_constraint:
            raise ValueError(
                f'If {name} is a sequence, the second value must be smaller'
                f'than {max_constraint}, not {max_degree}'
            )

        if type_constraint is not None:
            if not isinstance(min_degree, type_constraint) or \
                    not isinstance(max_degree, type_constraint):
                raise ValueError(
                    f'If {name} is a sequence, its values must be of'
                    f'type {type_constraint}, not {nums_range}'
                )
        return nums_range

    @staticmethod
    def parse_probability(probability: float) -> float:
        is_number = isinstance(probability, numbers.Number)
        if not (is_number and 0 <= probability <= 1):
            message = (
                'Probability must be a number in [0, 1],'
                f' not {probability}'
            )
            raise ValueError(message)
        return probability

    @staticmethod
    def parse_sample(sample: Subject) -> None:
        if not isinstance(sample, Subject):
            message = (
                'Input to a transform must be a tensor or an instance'
                f' of torchio.Subject, not "{type(sample)}"'
            )
            raise RuntimeError(message)

    def parse_tensor(self, data: TypeData) -> Subject:
        if data.ndim != 4:
            message = (
                'The input must be a 4D tensor with dimensions'
                f' (channels, x, y, z) but it has shape {tuple(data.shape)}'
            )
            raise ValueError(message)
        return self._get_subject_from_tensor(data)

    @staticmethod
    def parse_interpolation(interpolation: str) -> Interpolation:
        if isinstance(interpolation, Interpolation):
            message = (
                'Interpolation of type torchio.Interpolation'
                ' is deprecated, please use a string instead'
            )
            warnings.warn(message, FutureWarning)
        elif isinstance(interpolation, str):
            interpolation = interpolation.lower()
            supported_values = [key.name.lower() for key in Interpolation]
            if interpolation in supported_values:
                interpolation = getattr(Interpolation, interpolation.upper())
            else:
                message = (
                    f'Interpolation "{interpolation}" is not among'
                    f' the supported values: {supported_values}'
                )
                raise AttributeError(message)
        else:
            message = (
                'image_interpolation must be a string,'
                f' not {type(interpolation)}'
            )
            raise TypeError(message)
        return interpolation

    def _get_subject_from_tensor(self, tensor: torch.Tensor) -> Subject:
        image = ScalarImage(tensor=tensor, channels_last=False)
        return Subject({self.image_name: image})

    @staticmethod
    def _get_subject_from_image(image: Image) -> Subject:
        subject = Subject({IMAGE_NAME: image})
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

    @staticmethod
    def nib_to_sitk(data: TypeData, affine: TypeData) -> sitk.Image:
        return nib_to_sitk(data, affine)

    @staticmethod
    def sitk_to_nib(image: sitk.Image) -> Tuple[torch.Tensor, np.ndarray]:
        return sitk_to_nib(image)

    @property
    def name(self):
        return self.__class__.__name__
