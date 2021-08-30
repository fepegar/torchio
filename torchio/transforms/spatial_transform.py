from typing import List, Dict

from .transform import Transform
from ..data import Image
from ..data.subject import Subject


class SpatialTransform(Transform):
    """Transform that modifies image bounds or voxels positions."""

    def get_images(self, subject: Subject) -> List[Image]:
        images = subject.get_images(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        return images

    def get_images_dict(self, subject: Subject) -> Dict[str, Image]:
        images_dict = subject.get_images_dict(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        return images_dict
