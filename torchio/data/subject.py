import pprint
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)
from ..torchio import DATA
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
        ...     one_image=Image('path_to_image.nii.gz, torchio.INTENSITY),
        ...     a_segmentation=Image('path_to_seg.nii.gz, torchio.LABEL),
        ...     age=45,
        ...     name='John Doe',
        ...     hospital='Hospital Juan Negrín',
        ... )
        >>> # If you want to create the mapping before, or have spaces in the keys:
        >>> subject_dict = {
        ...     'one image': Image('path_to_image.nii.gz, torchio.INTENSITY),
        ...     'a segmentation': Image('path_to_seg.nii.gz, torchio.LABEL),
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

    def check_consistent_shape(self) -> None:
        shapes_dict = {}
        for key, image in self.items():
            if not isinstance(image, Image) or not image.is_sample:
                continue
            shapes_dict[key] = image[DATA].shape
        num_unique_shapes = len(set(shapes_dict.values()))
        if num_unique_shapes > 1:
            message = (
                'Images in sample have inconsistent shapes:'
                f'\n{pprint.pformat(shapes_dict)}'
            )
            raise ValueError(message)

    def add_transform(self, transform, parameters_dict):
        self.history.append((transform.name, parameters_dict))
