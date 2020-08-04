import copy
import collections
from typing import Dict, Sequence, Optional, Callable

from deprecated import deprecated
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
    :py:class:`torchio.data.subject.Subject` containing instances of
    :py:class:`torchio.data.image.Image`.
    The file format must be compatible with `NiBabel`_ or `SimpleITK`_ readers.
    It can also be a directory containing
    `DICOM`_ files.

    Indexing an :py:class:`~torchio.data.dataset.ImagesDataset` returns an
    instance of :py:class:`~torchio.data.subject.Subject`. Check out the
    documentation for both classes for usage examples.

    Example:

        >>> sample = images_dataset[0]
        >>> sample
        Subject(Keys: ('image', 'label'); images: 2)
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

    Example:
        >>> import torchio
        >>> from torchio import ImagesDataset, ScalarImage, LabelMap, Subject
        >>> from torchio.transforms import RescaleIntensity, RandomAffine, Compose
        >>> subject_a = Subject([
        ...     t1=ScalarImage('t1.nrrd',),
        ...     t2=ScalarImage('t2.mha',),
        ...     label=LabelMap('t1_seg.nii.gz'),
        ...     age=31,
        ...     name='Fernando Perez',
        >>> ])
        >>> subject_b = Subject(
        ...     t1=ScalarImage('colin27_t1_tal_lin.minc',),
        ...     t2=ScalarImage('colin27_t2_tal_lin_dicom',),
        ...     label=LabelMap('colin27_seg1.nii.gz'),
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
            ):
        self._parse_subjects_list(subjects)
        self.subjects = subjects
        self._transform: Optional[Callable]
        self.set_transform(transform)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index: int) -> dict:
        if not isinstance(index, int):
            raise ValueError(f'Index "{index}" must be int, not {type(index)}')
        subject = self.subjects[index]
        sample = copy.deepcopy(subject)  # cheap since images not loaded yet
        sample.load()

        # Apply transform (this is usually the bottleneck)
        if self._transform is not None:
            sample = self._transform(sample)
        return sample

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
    @deprecated(
        'ImagesDataset.save_sample is deprecated. Use Image.save instead'
    )
    def save_sample(
            cls,
            sample: Subject,
            output_paths_dict: Dict[str, TypePath],
            ) -> None:
        for key, output_path in output_paths_dict.items():
            tensor = sample[key][DATA]
            affine = sample[key][AFFINE]
            write_image(tensor, affine, output_path)
