import copy
import pprint
from typing import Any, Dict, List, Tuple, Optional, Sequence, TYPE_CHECKING

import numpy as np

from ..constants import TYPE, INTENSITY
from .image import Image
from ..utils import get_subclasses

if TYPE_CHECKING:
    from ..transforms import Transform, Compose


class Subject(dict):
    """Class to store information about the images corresponding to a subject.

    Args:
        *args: If provided, a dictionary of items.
        **kwargs: Items that will be added to the subject sample.

    Example:

        >>> import torchio as tio
        >>> # One way:
        >>> subject = tio.Subject(
        ...     one_image=tio.ScalarImage('path_to_image.nii.gz'),
        ...     a_segmentation=tio.LabelMap('path_to_seg.nii.gz'),
        ...     age=45,
        ...     name='John Doe',
        ...     hospital='Hospital Juan Negrín',
        ... )
        >>> # If you want to create the mapping before, or have spaces in the keys:
        >>> subject_dict = {
        ...     'one image': tio.ScalarImage('path_to_image.nii.gz'),
        ...     'a segmentation': tio.LabelMap('path_to_seg.nii.gz'),
        ...     'age': 45,
        ...     'name': 'John Doe',
        ...     'hospital': 'Hospital Juan Negrín',
        ... }
        >>> subject = tio.Subject(subject_dict)

    """

    def __init__(self, *args, **kwargs: Dict[str, Any]):
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                kwargs.update(args[0])
            else:
                message = (
                    'Only one dictionary as positional argument is allowed')
                raise ValueError(message)
        super().__init__(**kwargs)
        self._parse_images(self.get_images(intensity_only=False))
        self.update_attributes()  # this allows me to do e.g. subject.t1
        self.applied_transforms = []

    def __repr__(self):
        num_images = len(self.get_images(intensity_only=False))
        string = (
            f'{self.__class__.__name__}'
            f'(Keys: {tuple(self.keys())}; images: {num_images})'
        )
        return string

    def __copy__(self):
        result_dict = {}
        for key, value in self.items():
            if isinstance(value, Image):
                value = copy.copy(value)
            else:
                value = copy.deepcopy(value)
            result_dict[key] = value
        new = Subject(result_dict)
        new.applied_transforms = self.applied_transforms[:]
        return new

    def __len__(self):
        return len(self.get_images(intensity_only=False))

    @staticmethod
    def _parse_images(images: List[Tuple[str, Image]]) -> None:
        # Check that it's not empty
        if not images:
            raise ValueError('A subject without images cannot be created')

    @property
    def shape(self):
        """Return shape of first image in subject.

        Consistency of shapes across images in the subject is checked first.

        Example::

            >>> import torchio as tio
            >>> colin = tio.datasets.Colin27()
            >>> colin.shape
            (1, 181, 217, 181)

        """
        self.check_consistent_attribute('shape')
        return self.get_first_image().shape

    @property
    def spatial_shape(self):
        """Return spatial shape of first image in subject.

        Consistency of spatial shapes across images in the subject is checked
        first.

        Example::

            >>> import torchio as tio
            >>> colin = tio.datasets.Colin27()
            >>> colin.shape
            (181, 217, 181)
        """
        self.check_consistent_spatial_shape()
        return self.get_first_image().spatial_shape

    @property
    def spacing(self):
        """Return spacing of first image in subject.

        Consistency of spacings across images in the subject is checked first.

        Example::

            >>> import torchio as tio
            >>> colin = tio.datasets.Slicer()
            >>> colin.shape
            (1.0, 1.0, 1.2999954223632812)
        """
        self.check_consistent_attribute('spacing')
        return self.get_first_image().spacing

    @property
    def history(self):
        # Kept for backwards compatibility
        return self.get_applied_transforms()

    def is_2d(self):
        return all(i.is_2d() for i in self.get_images(intensity_only=False))

    def get_applied_transforms(
            self,
            ignore_intensity: bool = False,
            image_interpolation: Optional[str] = None,
            ) -> List['Transform']:
        from ..transforms.transform import Transform
        from ..transforms.intensity_transform import IntensityTransform
        name_to_transform = {
            cls.__name__: cls
            for cls in get_subclasses(Transform)
        }
        transforms_list = []
        for transform_name, arguments in self.applied_transforms:
            transform = name_to_transform[transform_name](**arguments)
            if ignore_intensity and isinstance(transform, IntensityTransform):
                continue
            resamples = hasattr(transform, 'image_interpolation')
            if resamples and image_interpolation is not None:
                parsed = transform.parse_interpolation(image_interpolation)
                transform.image_interpolation = parsed
            transforms_list.append(transform)
        return transforms_list

    def get_composed_history(
            self,
            ignore_intensity: bool = False,
            image_interpolation: Optional[str] = None,
            ) -> 'Compose':
        from ..transforms.augmentation.composition import Compose
        transforms = self.get_applied_transforms(
            ignore_intensity=ignore_intensity,
            image_interpolation=image_interpolation,
        )
        return Compose(transforms)

    def get_inverse_transform(
            self,
            warn: bool = True,
            ignore_intensity: bool = True,
            image_interpolation: Optional[str] = None,
            ) ->  'Compose':
        """Get a reversed list of the inverses of the applied transforms.

        Args:
            warn: Issue a warning if some transforms are not invertible.
            ignore_intensity: If ``True``, all instances of
                :class:`~torchio.transforms.intensity_transform.IntensityTransform`
                will be ignored.
            image_interpolation: Modify interpolation for scalar images inside
                transforms that perform resampling.
        """
        history_transform = self.get_composed_history(
            ignore_intensity=ignore_intensity,
            image_interpolation=image_interpolation,
        )
        inverse_transform = history_transform.inverse(warn=warn)
        return inverse_transform

    def apply_inverse_transform(self, **kwargs) -> 'Subject':
        """Try to apply the inverse of all applied transforms, in reverse order.

        Args:
            **kwargs: Keyword arguments passed on to
                :meth:`~torchio.data.subject.Subject.get_inverse_transform`.
        """
        inverse_transform = self.get_inverse_transform(**kwargs)
        transformed = inverse_transform(self)
        transformed.clear_history()
        return transformed

    def clear_history(self) -> None:
        self.applied_transforms = []

    def check_consistent_attribute(
            self,
            attribute: str,
            relative_tolerance: float = 1e-6,
            absolute_tolerance: float = 1e-6,
            message: Optional[str] = None,
            ) -> None:
        r"""Check for consistency of an attribute across all images.

        Args:
            attribute: Name of the image attribute to check
            relative_tolerance: Relative tolerance for :func:`numpy.allclose()`
            absolute_tolerance: Absolute tolerance for :func:`numpy.allclose()`

        Example:
            >>> import numpy as np
            >>> import torch
            >>> import torchio as tio
            >>> scalars = torch.randn(1, 512, 512, 100)
            >>> mask = torch.tensor(scalars > 0).type(torch.int16)
            >>> af1 = np.eye([0.8, 0.8, 2.50000000000001, 1])
            >>> af2 = np.eye([0.8, 0.8, 2.49999999999999, 1])  # small difference here (e.g. due to different reader)
            >>> subject = tio.Subject(
            ...   image = tio.ScalarImage(tensor=scalars, affine=af1),
            ...   mask = tio.LabelMap(tensor=mask, affine=af2)
            ... )
            >>> subject.check_consistent_attribute('spacing')  # no error as tolerances are > 0

        .. note:: To check that all values for a specific attribute are close
            between all images in the subject, :func:`numpy.allclose()` is used.
            This function returns ``True`` if
            :math:`|a_i - b_i| \leq t_{abs} + t_{rel} * |b_i|`, where
            :math:`a_i` and :math:`b_i` are the :math:`i`-th element of the same
            attribute of two images being compared,
            :math:`t_{abs}` is the ``absolute_tolerance`` and
            :math:`t_{rel}` is the ``relative_tolerance``.
        """
        message = (
            f'More than one value for "{attribute}" found in subject images:'
            '\n{}'
        )

        names_images = self.get_images_dict(intensity_only=False).items()
        try:
            first_attribute = None
            first_image = None

            for image_name, image in names_images:
                if first_attribute is None:
                    first_attribute = getattr(image, attribute)
                    first_image = image_name
                    continue
                current_attribute = getattr(image, attribute)
                all_close = np.allclose(
                    current_attribute,
                    first_attribute,
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                )
                if not all_close:
                    message = message.format(
                        pprint.pformat({
                            first_image: first_attribute,
                            image_name: current_attribute
                        }),
                    )
                    raise RuntimeError(message)
        except TypeError:
            # fallback for non-numeric values
            values_dict = {}
            for image_name, image in names_images:
                values_dict[image_name] = getattr(image, attribute)
            num_unique_values = len(set(values_dict.values()))
            if num_unique_values > 1:
                message = message.format(pprint.pformat(values_dict))
                raise RuntimeError(message)

    def check_consistent_spatial_shape(self) -> None:
        self.check_consistent_attribute('spatial_shape')

    def check_consistent_orientation(self) -> None:
        self.check_consistent_attribute('orientation')

    def check_consistent_affine(self):
        self.check_consistent_attribute('affine')

    def check_consistent_space(self):
        self.check_consistent_spatial_shape()
        self.check_consistent_affine()

    def get_images_names(self) -> List[str]:
        return list(self.get_images_dict(intensity_only=False).keys())

    def get_images_dict(
            self,
            intensity_only=True,
            include: Optional[Sequence[str]] = None,
            exclude: Optional[Sequence[str]] = None,
            ) -> Dict[str, Image]:
        images = {}
        for image_name, image in self.items():
            if not isinstance(image, Image):
                continue
            if intensity_only and not image[TYPE] == INTENSITY:
                continue
            if include is not None and image_name not in include:
                continue
            if exclude is not None and image_name in exclude:
                continue
            images[image_name] = image
        return images

    def get_images(
            self,
            intensity_only=True,
            include: Optional[Sequence[str]] = None,
            exclude: Optional[Sequence[str]] = None,
            ) -> List[Image]:
        images_dict = self.get_images_dict(
            intensity_only=intensity_only,
            include=include,
            exclude=exclude,
        )
        return list(images_dict.values())

    def get_first_image(self) -> Image:
        return self.get_images(intensity_only=False)[0]

    # flake8: noqa: F821
    def add_transform(
            self,
            transform: 'Transform',
            parameters_dict: dict,
            ) -> None:
        self.applied_transforms.append((transform.name, parameters_dict))

    def load(self) -> None:
        """Load images in subject on RAM."""
        for image in self.get_images(intensity_only=False):
            image.load()

    def update_attributes(self) -> None:
        # This allows to get images using attribute notation, e.g. subject.t1
        self.__dict__.update(self)

    def add_image(self, image: Image, image_name: str) -> None:
        """Add an image."""
        self[image_name] = image
        self.update_attributes()

    def remove_image(self, image_name: str) -> None:
        """Remove an image."""
        del self[image_name]
        delattr(self, image_name)

    def plot(self, **kwargs) -> None:
        """Plot images using matplotlib.

        Args:
            **kwargs: Keyword arguments that will be passed on to
                :meth:`~torchio.Image.plot`.
        """
        from ..visualization import plot_subject  # avoid circular import
        plot_subject(self, **kwargs)
