import copy
from typing import Sequence, Optional, Callable, Iterable

from torch.utils.data import Dataset

from .subject import Subject


class SubjectsDataset(Dataset):
    """Base TorchIO dataset.

    Reader of 3D medical images that directly inherits from the PyTorch
    :class:`~torch.utils.data.Dataset`. It can be used with a PyTorch
    :class:`~torch.utils.data.DataLoader` for efficient loading and
    augmentation. It receives a list of instances of :class:`~torchio.Subject`
    and an optional transform applied to the volumes after loading.

    Args:
        subjects: List of instances of :class:`~torchio.Subject`.
        transform: An instance of :class:`~torchio.transforms.Transform`
            that will be applied to each subject.
        load_getitem: Load all subject images before returning it in
            :meth:`__getitem__`. Set it to ``False`` if some of the images will
            not be needed during training.

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

    .. tip:: To quickly iterate over the subjects without loading the images,
        use :meth:`dry_iter()`.
    """  # noqa: E501

    def __init__(
            self,
            subjects: Sequence[Subject],
            transform: Optional[Callable] = None,
            load_getitem: bool = True,
            ):
        self._parse_subjects_list(subjects)
        self._subjects = subjects
        self._transform: Optional[Callable]
        self.set_transform(transform)
        self.load_getitem = load_getitem

    def __len__(self):
        return len(self._subjects)

    def __getitem__(self, index: int) -> Subject:
        if not isinstance(index, int):
            raise ValueError(f'Index "{index}" must be int, not {type(index)}')
        subject = self._subjects[index]
        subject = copy.deepcopy(subject)  # cheap since images not loaded yet
        if self.load_getitem:
            subject.load()

        # Apply transform (this is usually the bottleneck)
        if self._transform is not None:
            subject = self._transform(subject)
        return subject

    def dry_iter(self):
        """Return the internal list of subjects.

        This can be used to iterate over the subjects without loading the data
        and applying any transforms::

        >>> names = [subject.name for subject in dataset.dry_iter()]
        """
        return self._subjects

    def set_transform(self, transform: Optional[Callable]) -> None:
        """Set the :attr:`transform` attribute.

        Args:
            transform: Callable object, typically an subclass of
                :class:`torchio.transforms.Transform`.
        """
        if transform is not None and not callable(transform):
            message = (
                'The transform must be a callable object,'
                f' but it has type {type(transform)}'
            )
            raise ValueError(message)
        self._transform = transform

    @staticmethod
    def _parse_subjects_list(subjects_list: Iterable[Subject]) -> None:
        # Check that it's an iterable
        try:
            iter(subjects_list)
        except TypeError as e:
            message = (
                f'Subject list must be an iterable, not {type(subjects_list)}'
            )
            raise TypeError(message) from e

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
