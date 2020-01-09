import warnings
from pathlib import Path
from collections.abc import Sequence
import torch
from torch.utils.data import Dataset
from ..utils import get_stem
from ..io import read_image, write_image


class ImagesDataset(Dataset):
    def __init__(
            self,
            subjects_list,
            transform=None,
            check_nans=True,
            verbose=False,
            ):
        """
        Each element of subjects_list is a dictionary:
        subject_list = [
            Image('one_image', path_to_one_image, torchio.INTENSITY),
            Image('another_image', path_to_another_image, torchio.INTENSITY),
            Image('a_label', path_to_a_label, torchio.LABEL),
        }
        See examples/example_multimodal.py for -obviously- an example.
        """
        self.parse_subjects_list(subjects_list)
        self.subjects_list = subjects_list
        self.transform = transform
        self.check_nans = check_nans
        self.verbose = verbose

    def __len__(self):
        return len(self.subjects_list)

    def __getitem__(self, index):
        subject_images = self.subjects_list[index]
        sample = {}
        for image in subject_images:
            tensor, affine = image.load(check_nans=self.check_nans)
            image_dict = dict(
                data=tensor,
                path=str(image.path),
                affine=affine,
                stem=get_stem(image.path),
                type=image.type,
            )
            sample[image.name] = image_dict

        # Apply transform (this is usually the bottleneck)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def parse_subjects_list(subjects_list):
        if not isinstance(subjects_list, Sequence):
            raise TypeError(
                f'Subject list must be a sequence, not {type(subjects_list)}')
        if not subjects_list:
            raise ValueError('Subjects list is empty')
        for subject_images in subjects_list:
            if not isinstance(subject_images, Sequence):
                message = (
                    'Subject images list must be a sequence'
                    f', not {type(subject_images)}'
                )
                raise TypeError(message)
            for image in subject_images:
                if not isinstance(image, Image):
                    message = (
                        'Subject list elements must be instances of'
                        f' torchio.Image, not {type(image)}'
                    )
                    raise TypeError(message)

    @staticmethod
    def save_sample(sample, output_paths_dict):
        for key, output_path in output_paths_dict.items():
            tensor = sample[key]['data'][0]  # remove channels dim
            affine = sample[key]['affine']
            write_image(tensor, affine, output_path)


class Image:
    def __init__(self, name, path, type_):
        self.name = name
        self.path = self.parse_path(path)
        self.type = type_

    def parse_path(self, path):
        try:
            path = Path(path).expanduser()
        except TypeError:
            print(f'Conversion to path not possible for variable: {path}')
            raise
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
