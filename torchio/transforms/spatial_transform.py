from typing import List

from .transform import Transform
from ..data import Image


class SpatialTransform(Transform):
    """Transform that modifies image bounds or voxels positions."""

    def get_images(self, sample) -> List[Image]:
        return sample.get_images(intensity_only=False, include=self.include, exclude=self.exclude)
