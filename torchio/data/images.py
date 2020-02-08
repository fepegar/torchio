import typing
import warnings
import collections
from pathlib import Path
from typing import Union, Sequence, Optional, Any, TypeVar, Dict, List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
from ..utils import get_stem
from ..torchio import DATA, AFFINE, TypePath
from .io import read_image, write_image


class Image:
    def __init__(self, name: str, path: TypePath, type_: str):
        self.name = name
        self.path = self.parse_path(path)
        self.type = type_

    def parse_path(self, path: TypePath) -> Path:
        try:
            path = Path(path).expanduser()
        except TypeError:
            message = f'Conversion to path not possible for variable: {path}'
            raise TypeError(message)
        if not path.is_file():
            message = (
                f'File for image "{self.name}"'
                f' not found: "{path}"'
                )
            raise FileNotFoundError(message)
        return path

    def load(self, check_nans: bool = True) -> Tuple[torch.Tensor, np.ndarray]:
        tensor, affine = read_image(self.path)
        tensor = tensor.unsqueeze(0)  # add channels dimension
        if check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in file "{self.path}"')
        return tensor, affine


class Subject(list):
    def __init__(self, *images: Image, name: Optional[str] = None):
        self.parse_images(images)
        super().__init__(images)
        self.name = name

    @staticmethod
    def parse_images(images: Sequence[Image]) -> None:
        # Check that each element is a list
        if not isinstance(images, collections.abc.Sequence):
            message = (
                'Subject "images" parameter must be a sequence'
                f', not {type(images)}'
            )
            raise TypeError(message)

        # Check that it's not empty
        if not images:
            raise ValueError('Images list is empty')

        # Check that there are only instances of Image
        # and all images have different names
        names: List[str] = []
        for image in images:
            if not isinstance(image, Image):
                message = (
                    'Subject list elements must be instances of'
                    f' torchio.Image, not {type(image)}'
                )
                raise TypeError(message)
            if image.name in names:
                message = (
                    f'More than one image with name "{image.name}"'
                    ' found in images list'
                )
                raise KeyError(message)
            names.append(image.name)


class ImagesDataset(Dataset):
    def __init__(
            self,
            subjects: Sequence[Subject],
            transform: Optional[Any] = None,
            check_nans: bool = True,
            verbose: bool = False,
            ):
        self.parse_subjects_list(subjects)
        self.subjects = subjects
        self._transform = transform
        self.check_nans = check_nans
        self.verbose = verbose

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index: int) -> dict:
        subject = self.subjects[index]
        sample = {}
        for image in subject:
            tensor, affine = image.load(check_nans=self.check_nans)
            image_dict = {
                DATA: tensor,
                AFFINE: affine,
                'type': image.type,
                'path': str(image.path),
                'stem': get_stem(image.path),
            }
            sample[image.name] = image_dict

        # Apply transform (this is usually the bottleneck)
        if self._transform is not None:
            sample = self._transform(sample)
        return sample

    def set_transform(self, transform: Any) -> None:
        self._transform = transform

    @staticmethod
    def parse_subjects_list(subjects_list: Sequence[Subject]) -> None:
        # Check that it's list or tuple
        if not isinstance(subjects_list, collections.abc.Sequence):
            raise TypeError(
                f'Subject list must be a sequence, not {type(subjects_list)}')

        # Check that it's not empty
        if not subjects_list:
            raise ValueError('Subjects list is empty')

        # Check each element
        for subject_list in subjects_list:
            Subject(*subject_list)

    @classmethod
    def save_sample(
            cls,
            sample: Dict[str, dict],
            output_paths_dict: Dict[str, TypePath],
            ) -> None:
        for key, output_path in output_paths_dict.items():
            tensor = sample[key][DATA][0]  # remove channels dim
            affine = sample[key][AFFINE]
            write_image(tensor, affine, output_path)
