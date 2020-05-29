import pprint
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)
from ..torchio import TYPE, INTENSITY
from .image import Image


class Subject(dict):
    """Class to store information about the images corresponding to a subject.

    Args:
        *args: If provided, a dictionary of items.
        **kwargs: Items that will be added to the subject sample.

    Example:

        >>> import torchio
        >>> from torchio import Image, Subject
        >>> # One way:
        >>> subject = Subject(
        ...     one_image=Image('path_to_image.nii.gz', torchio.INTENSITY),
        ...     a_segmentation=Image('path_to_seg.nii.gz', torchio.LABEL),
        ...     age=45,
        ...     name='John Doe',
        ...     hospital='Hospital Juan Negrín',
        ... )
        >>> # If you want to create the mapping before, or have spaces in the keys:
        >>> subject_dict = {
        ...     'one image': Image('path_to_image.nii.gz', torchio.INTENSITY),
        ...     'a segmentation': Image('path_to_seg.nii.gz', torchio.LABEL),
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
        self.images = [
            (k, v) for (k, v) in self.items()
            if isinstance(v, Image)
        ]
        self._parse_images(self.images)
        self.is_sample = False  # set to True by ImagesDataset
        self.history = []

    def __repr__(self):
        string = (
            f'{self.__class__.__name__}'
            f'(Keys: {tuple(self.keys())}; images: {len(self.images)})'
        )
        return string

    @staticmethod
    def _parse_images(images: List[Tuple[str, Image]]) -> None:
        # Check that it's not empty
        if not images:
            raise ValueError('A subject without images cannot be created')

    @property
    def shape(self):
        """Return shape of first image in sample.

        Consistency of shapes across images in the sample is checked first.
        """
        self.check_consistent_shape()
        image = self.get_images(intensity_only=False)[0]
        return image.shape

    @property
    def spatial_shape(self):
        """Return spatial shape of first image in sample.

        Consistency of shapes across images in the sample is checked first.
        """
        return self.shape[1:]

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

    def check_consistent_shape(self) -> None:
        shapes_dict = {}
        iterable = self.get_images_dict(intensity_only=False).items()
        for image_name, image in iterable:
            shapes_dict[image_name] = image.shape
        num_unique_shapes = len(set(shapes_dict.values()))
        if num_unique_shapes > 1:
            message = (
                'Images in sample have inconsistent shapes:'
                f'\n{pprint.pformat(shapes_dict)}'
            )
            raise ValueError(message)

    def add_transform(
            self,
            transform: 'Transform',
            parameters_dict: dict,
            ) -> None:
        self.history.append((transform.name, parameters_dict))
