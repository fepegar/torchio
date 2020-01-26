import warnings
from pathlib import Path
from collections.abc import Sequence
import torch
from torch.utils.data import Dataset
from ..utils import get_stem
from ..torchio import DATA, AFFINE
from .io import read_image, write_image


class ImagesDataset(Dataset):
    def __init__(
            self,
            subjects_list,
            transform=None,
            check_nans=True,
            verbose=False,
            ):
        """
        Each element of subjects_list should be an instance of torchio.Subject
        """
        self.parse_subjects_list(subjects_list)
        self.subjects_list = subjects_list
        self._transform = transform
        self.check_nans = check_nans
        self.verbose = verbose

    def __len__(self):
        return len(self.subjects_list)

    def __getitem__(self, index):
        subject_images = self.subjects_list[index]
        sample = {}
        for image in subject_images:
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

    def set_transform(self, transform):
        self._transform = transform

    @staticmethod
    def parse_subjects_list(subjects_list):
        # Check that it's list or tuple
        if not isinstance(subjects_list, Sequence):
            raise TypeError(
                f'Subject list must be a sequence, not {type(subjects_list)}')

        # Check that it's not empty
        if not subjects_list:
            raise ValueError('Subjects list is empty')

        # Check each element
        for subject_list in subjects_list:
            subject = Subject(*subject_list)

    @classmethod
    def save_sample(cls, sample, output_paths_dict):
        for key, output_path in output_paths_dict.items():
            tensor = sample[key][DATA][0]  # remove channels dim
            affine = sample[key][AFFINE]
            write_image(tensor, affine, output_path)


class Subject(list):
    def __init__(self, *images, name=None):
        self.parse_images(images)
        super().__init__(images)
        self.name = name

    @staticmethod
    def parse_images(images):
        # Check that each element is a list
        if not isinstance(images, Sequence):
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
        names = []
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


class Image:
    def __init__(self, name, path, type_):
        self.name = name
        self.path = self.parse_path(path)
        self.type = type_

    def parse_path(self, path):
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

    def load(self, check_nans=True):
        tensor, affine = read_image(self.path)
        tensor = tensor.unsqueeze(0)  # add channels dimension
        if check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in file "{self.path}"')
        return tensor, affine
