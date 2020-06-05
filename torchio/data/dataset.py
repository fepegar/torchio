import copy
import collections
from typing import (
    Dict,
    Sequence,
    Optional,
    Callable,
)
from torch.utils.data import Dataset
from ..utils import get_stem
from ..torchio import DATA, AFFINE, TYPE, PATH, STEM, TypePath
from .image import Image
from .io import write_image
from .subject import Subject


class ImagesDataset(Dataset):
    """Base TorchIO dataset.

    :py:class:`~torchio.data.dataset.ImagesDataset`
    is a reader of 3D medical images that directly
    inherits from :class:`torch.utils.data.Dataset`.
    It can be used with a :class:`torch.utils.data.DataLoader`
    for efficient loading and augmentation.
    It receives a list of subjects, where each subject is an instance of
    :py:class:`~torchio.data.subject.Subject` containing instances of
    :py:class:`~torchio.data.image.Image`.
    The file format must be compatible with `NiBabel`_ or `SimpleITK`_ readers.
    It can also be a directory containing
    `DICOM`_ files.

    Indexing an :py:class:`~torchio.data.dataset.ImagesDataset` returns a
    Python dictionary with the data corresponding to the queried subject.
    The keys in the dictionary are the names of the images passed to that
    subject, for example ``('t1', 't2', 'segmentation')``.

    The value corresponding to each image name is typically an instance of
    :py:class:`~torchio.data.image.Image` with information about the image.
    The data is stored in ``image[torchio.DATA]`` (or just ``image.data``),
    and the corresponding `affine matrix`_
    is in ``image[torchio.AFFINE]`` (or just ``image.affine``):

        >>> sample = images_dataset[0]
        >>> sample.keys()
        dict_keys(['image', 'label'])
        >>> image = sample['image']  # or sample.image
        >>> image.shape
        torch.Size([1, 176, 256, 256])
        >>> image.affine
        array([[   0.03,    1.13,   -0.08,  -88.54],
               [   0.06,    0.08,    0.95, -129.66],
               [   1.18,   -0.06,   -0.11,  -67.15],
               [   0.  ,    0.  ,    0.  ,    1.  ]])

    Args:
        subjects: Sequence of instances of
            :class:`~torchio.data.subject.Subject`.
        transform: An instance of :py:class:`torchio.transforms.Transform`
            that will be applied to each sample.
        check_nans: If ``True``, issues a warning if NaNs are found
            in the image.
        load_image_data: If ``False``, image data and affine will not be loaded.
            These fields will be set to ``None`` in the sample. This can be
            used to quickly iterate over the samples to retrieve e.g. the
            images paths. If ``True``, transform must be ``None``.

    Example:
        >>> import torchio
        >>> from torchio import ImagesDataset, Image, Subject
        >>> from torchio.transforms import RescaleIntensity, RandomAffine, Compose
        >>> subject_a = Subject([
        ...     t1=Image('~/Dropbox/MRI/t1.nrrd', torchio.INTENSITY),
        ...     t2=Image('~/Dropbox/MRI/t2.mha', torchio.INTENSITY),
        ...     label=Image('~/Dropbox/MRI/t1_seg.nii.gz', torchio.LABEL),
        ...     age=31,
        ...     name='Fernando Perez',
        >>> ])
        >>> subject_b = Subject(
        ...     t1=Image('/tmp/colin27_t1_tal_lin.nii.gz', torchio.INTENSITY),
        ...     t2=Image('/tmp/colin27_t2_tal_lin.nii', torchio.INTENSITY),
        ...     label=Image('/tmp/colin27_seg1.nii.gz', torchio.LABEL),
        ...     age=56,
        ...     name='Colin Holmes',
        ... )
        >>> subjects_list = [subject_a, subject_b]
        >>> transforms = [
        ...     RescaleIntensity((0, 1)),
        ...     RandomAffine(),
        ... ]
        >>> transform = Compose(transforms)
        >>> subjects_dataset = ImagesDataset(subjects_list, transform=transform)
        >>> subject_sample = subjects_dataset[0]

    .. _NiBabel: https://nipy.org/nibabel/#nibabel
    .. _SimpleITK: https://itk.org/Wiki/ITK/FAQ#What_3D_file_formats_can_ITK_import_and_export.3F
    .. _DICOM: https://www.dicomstandard.org/
    .. _affine matrix: https://nipy.org/nibabel/coordinate_systems.html

    """

    def __init__(
            self,
            subjects: Sequence[Subject],
            transform: Optional[Callable] = None,
            check_nans: bool = True,
            load_image_data: bool = True,
            ):
        self._parse_subjects_list(subjects)
        self.subjects = subjects
        self._transform: Optional[Callable]
        self.set_transform(transform)
        self.check_nans = check_nans
        self._load_image_data: bool
        self.set_load_image_data(load_image_data)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, int):
            raise ValueError(f'Index "{index}" must be int, not {type(index)}')
        subject = self.subjects[index]
        sample = self._get_sample_dict_from_subject(subject)

        # Apply transform (this is usually the bottleneck)
        if self._transform is not None:
            sample = self._transform(sample)
        return sample

    def _get_sample_dict_from_subject(self, subject: Subject):
        """Create a dictionary of dictionaries with subject information.

        Args:
            subject: Instance of :py:class:`~torchio.data.subject.Subject`.
        """
        subject_sample = copy.deepcopy(subject)
        for (key, value) in subject.items():
            if isinstance(value, Image):
                value = self._get_image_dict_from_image(value)
            subject_sample[key] = value
        # This allows me to do e.g. subject.t1
        subject_sample.__dict__.update(subject_sample)
        subject_sample.is_sample = True
        return subject_sample

    def _get_image_dict_from_image(self, image: Image):
        """Create a dictionary with image information.

        Args:
            image: Instance of :py:class:`~torchio.data.dataset.Image`.

        Return:
            Dictionary with keys
            :py:attr:`torchio.DATA`,
            :py:attr:`torchio.AFFINE`,
            :py:attr:`torchio.TYPE`,
            :py:attr:`torchio.PATH` and
            :py:attr:`torchio.STEM`.
        """
        if self._load_image_data:
            tensor, affine = image.load(check_nans=self.check_nans)
        else:
            tensor = affine = None
        path = '' if image.path is None else str(image.path)
        stem = '' if image.path is None else get_stem(image.path)
        image_dict = {
            DATA: tensor,
            AFFINE: affine,
            TYPE: image.type,
            PATH: path,
            STEM: stem,
        }
        image = copy.deepcopy(image)
        image.update(image_dict)
        image.is_sample = True
        return image

    def set_transform(self, transform: Optional[Callable]) -> None:
        """Set the :attr:`transform` attribute.

        Args:
            transform: An instance of :py:class:`torchio.transforms.Transform`.
        """
        if transform is not None and not callable(transform):
            raise ValueError(
                f'The transform must be a callable object, not {transform}')
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
        for subject in subjects_list:
            if not isinstance(subject, Subject):
                message = (
                    'Subjects list must contain instances of torchio.Subject,'
                    f' not "{type(subject)}"'
                )
                raise TypeError(message)

    @classmethod
    def save_sample(
            cls,
            sample: Subject,
            output_paths_dict: Dict[str, TypePath],
            ) -> None:
        for key, output_path in output_paths_dict.items():
            tensor = sample[key][DATA].squeeze()  # assume 2D if (1, 1, H, W)
            affine = sample[key][AFFINE]
            write_image(tensor, affine, output_path)

    def set_load_image_data(self, load_image_data: bool):
        if not load_image_data and self._transform is not None:
            message = (
                'Load data cannot be set to False if transform is not None.'
                f'Current transform is {self._transform}')
            raise ValueError(message)
        self._load_image_data = load_image_data
