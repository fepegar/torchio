import copy
import collections
from typing import Sequence, Optional, Callable

from torch.utils.data import Dataset

from .subject import Subject


class SubjectsDataset(Dataset):
    """Base TorchIO dataset.

    :class:`~torchio.data.dataset.SubjectsDataset`
    is a reader of 3D medical images that directly
    inherits from :class:`torch.utils.data.Dataset`.
    It can be used with a :class:`torch.utils.data.DataLoader`
    for efficient loading and augmentation.
    It receives a list of instances of
    :class:`torchio.data.subject.Subject`.

    Args:
        subjects: List of instances of
            :class:`~torchio.data.subject.Subject`.
        transform: An instance of :class:`torchio.transforms.Transform`
            that will be applied to each subject.

    Example:
        >>> import torchio as tio
        >>> subject_a = tio.Subject(
        ...     t1=tio.ScalarImage('t1.nrrd',),
        ...     t2=tio.ScalarImage('t2.mha',),
        ...     label=tio.LabelMap('t1_seg.nii.gz'),
        ...     age=31,
        ...     name='Fernando Perez',
        ... )
        >>> subject_b = tio.Subject(
        ...     t1=tio.ScalarImage('colin27_t1_tal_lin.minc',),
        ...     t2=tio.ScalarImage('colin27_t2_tal_lin_dicom',),
        ...     label=tio.LabelMap('colin27_seg1.nii.gz'),
        ...     age=56,
        ...     name='Colin Holmes',
        ... )
        >>> subjects_list = [subject_a, subject_b]
        >>> transforms = [
        ...     tio.RescaleIntensity((0, 1)),
        ...     tio.RandomAffine(),
        ... ]
        >>> transform = tio.Compose(transforms)
        >>> subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)
        >>> subject = subjects_dataset[0]

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

    def __getitem__(self, index: int) -> Subject:
        if not isinstance(index, int):
            raise ValueError(f'Index "{index}" must be int, not {type(index)}')
        subject = self.subjects[index]
        subject = copy.deepcopy(subject)  # cheap since images not loaded yet
        subject.load()

        # Apply transform (this is usually the bottleneck)
        if self._transform is not None:
            subject = self._transform(subject)
        return subject

    def set_transform(self, transform: Optional[Callable]) -> None:
        """Set the :attr:`transform` attribute.

        Args:
            transform: An instance of :class:`torchio.transforms.Transform`.
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
