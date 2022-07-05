from typing import List

from ..data import Image
from ..data.subject import Subject
from .transform import Transform


class SpatialTransform(Transform):
    """Transform that modifies image bounds or voxels positions."""

    def get_images(self, subject: Subject) -> List[Image]:
        images = subject.get_images(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        return images
