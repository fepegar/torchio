from typing import Tuple, Optional, Generator

import numpy as np

from ... import TypePatchSize
from ...data.subject import Subject
from ...utils import to_tuple


class PatchSampler:
    r"""Base class for TorchIO samplers.

    Args:
        patch_size: Tuple of integers :math:`(d, h, w)` to generate patches
            of size :math:`d \times h \times w`.
            If a single number :math:`n` is provided, :math:`d = h = w = n`.
    """
    def __init__(self, patch_size: TypePatchSize):
        patch_size_array = np.array(to_tuple(patch_size, length=3))
        if np.any(patch_size_array < 1):
            message = (
                'Patch dimensions must be positive integers,'
                f' not {patch_size_array}'
            )
            raise ValueError(message)
        self.patch_size = patch_size_array.astype(np.uint16)

    def extract_patch(self):
        raise NotImplementedError

    @staticmethod
    def get_crop_transform(
            image_size,
            index_ini,
            patch_size: TypePatchSize,
            ):
        from ...transforms.preprocessing.spatial.crop import Crop
        image_size = np.array(image_size, dtype=np.uint16)
        index_ini = np.array(index_ini, dtype=np.uint16)
        patch_size = np.array(patch_size, dtype=np.uint16)
        index_fin = index_ini + patch_size
        crop_ini = index_ini.tolist()
        crop_fin = (image_size - index_fin).tolist()
        TypeBounds = Tuple[int, int, int, int, int, int]
        start = ()
        cropping: TypeBounds = sum(zip(crop_ini, crop_fin), start)
        return Crop(cropping)


class RandomSampler(PatchSampler):
    r"""Base class for TorchIO samplers.

    Args:
        patch_size: Tuple of integers :math:`(d, h, w)` to generate patches
            of size :math:`d \times h \times w`.
            If a single number :math:`n` is provided, :math:`d = h = w = n`.
    """
    def __call__(
            self,
            sample: Subject,
            num_patches: Optional[int] = None,
            ) -> Generator[Subject, None, None]:
        raise NotImplementedError

    def get_probability_map(self, sample: Subject):
        raise NotImplementedError
