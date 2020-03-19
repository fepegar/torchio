import typing
import warnings
import collections
from pathlib import Path
from typing import Union, Sequence, Optional, Any, TypeVar, Dict, List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
from ..utils import get_stem
from ..torchio import DATA, AFFINE, TYPE, PATH, STEM, TypePath
from .io import read_image, write_image


class Image:
    r"""Class to store information about an image.

    Args:
        name: String corresponding to the name of the image, e.g. ``t1``,
            or ``segmentation``.
        path: Path to a file that can be read by
            :mod:`SimpleITK` or :mod:`nibabel` or to a directory containing
            DICOM files.
        type\_: Type of image, such as :attr:`torchio.INTENSITY` or
            :attr:`torchio.LABEL`. This will be used by the transforms to
            decide whether to apply an operation, or which interpolation to use
            when resampling.
    """
    def __init__(self, name: str, path: TypePath, type_: str):
        self.name = name
        self.path = self._parse_path(path)
        self.type = type_

    def _parse_path(self, path: TypePath) -> Path:
        try:
            path = Path(path).expanduser()
        except TypeError:
            message = f'Conversion to path not possible for variable: {path}'
            raise TypeError(message)
        if not (path.is_file() or path.is_dir()):  # might be a dir with DICOM
            message = (
                f'File for image "{self.name}"'
                f' not found: "{path}"'
                )
            raise FileNotFoundError(message)
        return path

    def load(self, check_nans: bool = True) -> Tuple[torch.Tensor, np.ndarray]:
        r"""Load the image from disk.

        The file is expected to be monomodal and 3D. A channels dimension is
        added to the tensor.

        Args:
            check_nans: If ``True``, issues a warning if NaNs are found
                in the image

        Returns:
            Tuple containing a 4D data tensor of size
            :math:`(1, D_{in}, H_{in}, W_{in})`
            and a 2D 4x4 affine matrix
        """
        tensor, affine = read_image(self.path)
        tensor = tensor.unsqueeze(0)  # add channels dimension
        if check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in file "{self.path}"')
        return tensor, affine


class Subject(list):
    """Class to store information about the images corresponding to a subject.

    Args:
        *images: Instances of :class:`~torchio.data.images.Image`.
        name: Subject ID
    """
    def __init__(self, *images: Image, name: str = ''):
        self._parse_images(images)
        super().__init__(images)
        self.name = name

    def __repr__(self):
        return f'{__class__.__name__}("{self.name}", {len(self)} images)'

    @staticmethod
    def _parse_images(images: Tuple[Image]) -> None:
        # Check that it's not empty
        if not images:
            raise ValueError('A subject without images cannot be created')

        # Check that there are only instances of Image
        # and all images have different names
        names: List[str] = []
        for image in images:
            if not isinstance(image, Image):
                message = (
                    'Subject list elements must be instances of'
                    f' torchio.Image, not "{type(image)}"'
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
    """Base TorchIO dataset.

    :class:`~torchio.data.images.ImagesDataset`
    is a reader of 3D medical images that directly
    inherits from :class:`torch.utils.data.Dataset`.
    It can be used with a :class:`torch.utils.data.DataLoader`
    for efficient loading and augmentation.
    It receives a list of subjects, where each subject is an instance of
    :class:`~torchio.data.images.Subject` containing instances of
    :class:`~torchio.data.images.Image`.
    The file format must be compatible with `NiBabel`_ or `SimpleITK`_ readers.
    It can also be a directory containing
    `DICOM`_ files.

    Indexing an :class:`~torchio.data.images.ImagesDataset` returns a
    Python dictionary with the data corresponding to the queried subject.
    The keys in the dictionary are the names of the images passed to that
    subject, for example ``('t1', 't2', 'segmentation')``.

    The value corresponding to each image name is another dictionary
    ``image_dict`` with information about the image.
    The data is stored in ``image_dict[torchio.IMAGE]``,
    and the corresponding `affine matrix`_
    is in ``image_dict[torchio.AFFINE]``:

        >>> sample = images_dataset[0]
        >>> sample.keys()
        dict_keys(['image', 'label'])
        >>> image_dict = sample['image']
        >>> image_dict[torchio.DATA].shape
        torch.Size([1, 176, 256, 256])
        >>> image_dict[torchio.AFFINE]
        array([[   0.03,    1.13,   -0.08,  -88.54],
               [   0.06,    0.08,    0.95, -129.66],
               [   1.18,   -0.06,   -0.11,  -67.15],
               [   0.  ,    0.  ,    0.  ,    1.  ]])

    Args:
        subjects: Sequence of instances of
            :class:`~torchio.data.images.Subject`.
        transform: An instance of
            :class:`torchio.transforms.Transform` that is applied to each
            image after loading it.
        check_nans: If ``True``, issues a warning if NaNs are found
            in the image

    .. _NiBabel: https://nipy.org/nibabel/#nibabel
    .. _SimpleITK: https://itk.org/Wiki/ITK/FAQ#What_3D_file_formats_can_ITK_import_and_export.3F
    .. _DICOM: https://www.dicomstandard.org/
    .. _affine matrix: https://nipy.org/nibabel/coordinate_systems.html

    """
    def __init__(
            self,
            subjects: Sequence[Subject],
            transform: Optional[Any] = None,
            check_nans: bool = True,
            ):
        self._parse_subjects_list(subjects)
        self.subjects = subjects
        self._transform = transform
        self.check_nans = check_nans

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, int):
            raise TypeError(f'Index "{index}" must be int, not {type(index)}')
        subject = self.subjects[index]
        sample = {}
        for image in subject:
            tensor, affine = image.load(check_nans=self.check_nans)
            image_dict = {
                DATA: tensor,
                AFFINE: affine,
                TYPE: image.type,
                PATH: str(image.path),
                STEM: get_stem(image.path),
            }
            sample[image.name] = image_dict

        # Apply transform (this is usually the bottleneck)
        if self._transform is not None:
            sample = self._transform(sample)
        return sample

    def set_transform(self, transform: Any) -> None:
        """Set the :attr:`transform` attribute.

        Args:
            transform: An instance of :class:`torchio.transforms.Transform`
        """
        self._transform = transform

    @staticmethod
    def _parse_subjects_list(subjects_list: Sequence[Subject]) -> None:
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
