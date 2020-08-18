import copy
import pprint
from typing import Any, Dict, List, Tuple
from ..torchio import TYPE, INTENSITY
from .image import Image


class Subject(dict):
    """Class to store information about the images corresponding to a subject.

    Args:
        *args: If provided, a dictionary of items.
        **kwargs: Items that will be added to the subject sample.

    Example:

        >>> import torchio
        >>> from torchio import ScalarImage, LabelMap, Subject
        >>> # One way:
        >>> subject = Subject(
        ...     one_image=ScalarImage('path_to_image.nii.gz'),
        ...     a_segmentation=LabelMap('path_to_seg.nii.gz'),
        ...     age=45,
        ...     name='John Doe',
        ...     hospital='Hospital Juan Negrín',
        ... )
        >>> # If you want to create the mapping before, or have spaces in the keys:
        >>> subject_dict = {
        ...     'one image': ScalarImage('path_to_image.nii.gz'),
        ...     'a segmentation': LabelMap('path_to_seg.nii.gz'),
        ...     'age': 45,
        ...     'name': 'John Doe',
        ...     'hospital': 'Hospital Juan Negrín',
        ... }
        >>> Subject(subject_dict)

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
        self.history = []

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
        new.history = self.history[:]
        return new

    @staticmethod
    def _parse_images(images: List[Tuple[str, Image]]) -> None:
        # Check that it's not empty
        if not images:
            raise ValueError('A subject without images cannot be created')

    @property
    def shape(self):
        """Return shape of first image in subject.

        Consistency of shapes across images in the subject is checked first.
        """
        self.check_consistent_attribute('shape')
        return self.get_first_image().shape

    @property
    def spatial_shape(self):
        """Return spatial shape of first image in subject.

        Consistency of spatial shapes across images in the subject is checked
        first.
        """
        self.check_consistent_spatial_shape()
        return self.get_first_image().spatial_shape

    @property
    def spacing(self):
        """Return spacing of first image in subject.

        Consistency of spacings across images in the subject is checked first.
        """
        self.check_consistent_attribute('spacing')
        return self.get_first_image().spacing

    def check_consistent_attribute(self, attribute: str) -> None:
        values_dict = {}
        iterable = self.get_images_dict(intensity_only=False).items()
        for image_name, image in iterable:
            values_dict[image_name] = getattr(image, attribute)
        num_unique_values = len(set(values_dict.values()))
        if num_unique_values > 1:
            message = (
                f'More than one {attribute} found in subject images:'
                f'\n{pprint.pformat(values_dict)}'
            )
            raise RuntimeError(message)

    def check_consistent_spatial_shape(self) -> None:
        self.check_consistent_attribute('spatial_shape')

    def get_images_dict(self, intensity_only=True):
        images = {}
        for image_name, image in self.items():
            if not isinstance(image, Image):
                continue
            if intensity_only and not image[TYPE] == INTENSITY:
                continue
            images[image_name] = image
        return images

    def get_images(self, intensity_only=True):
        images_dict = self.get_images_dict(intensity_only=intensity_only)
        return list(images_dict.values())

    def get_first_image(self):
        return self.get_images(intensity_only=False)[0]

    # flake8: noqa: F821
    def add_transform(
            self,
            transform: 'Transform',
            parameters_dict: dict,
            ) -> None:
        self.history.append((transform.name, parameters_dict))

    def load(self):
        for image in self.get_images(intensity_only=False):
            image.load()

    def crop(self, index_ini, index_fin):
        result_dict = {}
        for key, value in self.items():
            if isinstance(value, Image):
                # patch.clone() is much faster than copy.deepcopy(patch)
                value = value.crop(index_ini, index_fin)
            else:
                value = copy.deepcopy(value)
            result_dict[key] = value
        new = Subject(result_dict)
        new.history = self.history
        return new

    def update_attributes(self):
        # This allows to get images using attribute notation, e.g. subject.t1
        self.__dict__.update(self)

    def add_image(self, image: Image, image_name: str) -> None:
        self[image_name] = image
        self.update_attributes()

    def remove_image(self, image_name: str) -> None:
        del self[image_name]
