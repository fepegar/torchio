from typing import Optional, Generator

import numpy as np

from ...typing import TypePatchSize, TypeTripletInt
from ...data.subject import Subject
from ...utils import to_tuple


class PatchSampler:
    r"""Base class for TorchIO samplers.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.

    .. warning:: This is an abstract class that should only be instantiated
        using child classes such as :class:`~torchio.data.UniformSampler` and
        :class:`~torchio.data.WeightedSampler`.
    """
    def __init__(self, patch_size: TypePatchSize):
        patch_size_array = np.array(to_tuple(patch_size, length=3))
        for n in patch_size_array:
            if n < 1 or not isinstance(n, (int, np.integer)):
                message = (
                    'Patch dimensions must be positive integers,'
                    f' not {patch_size_array}'
                )
                raise ValueError(message)
        self.patch_size = patch_size_array.astype(np.uint16)

    def extract_patch(
            self,
            subject: Subject,
            index_ini: TypeTripletInt,
            ) -> Subject:
        cropped_subject = self.crop(subject, index_ini, self.patch_size)
        cropped_subject['index_ini'] = np.array(index_ini).astype(int)
        return cropped_subject

    def crop(
            self,
            subject: Subject,
            index_ini: TypeTripletInt,
            patch_size: TypeTripletInt,
            ) -> Subject:
        transform = self.get_crop_transform(subject, index_ini, patch_size)
        return transform(subject)

    @staticmethod
    def get_crop_transform(
            subject,
            index_ini,
            patch_size: TypePatchSize,
            ):
        from ...transforms.preprocessing.spatial.crop import Crop
        shape = np.array(subject.spatial_shape, dtype=np.uint16)
        index_ini = np.array(index_ini, dtype=np.uint16)
        patch_size = np.array(patch_size, dtype=np.uint16)
        index_fin = index_ini + patch_size
        crop_ini = index_ini.tolist()
        crop_fin = (shape - index_fin).tolist()
        start = ()
        cropping = sum(zip(crop_ini, crop_fin), start)
        return Crop(cropping)


class RandomSampler(PatchSampler):
    r"""Base class for random samplers.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.
    """
    def __call__(
            self,
            subject: Subject,
            num_patches: Optional[int] = None,
            ) -> Generator[Subject, None, None]:
        raise NotImplementedError

    def get_probability_map(self, subject: Subject):
        raise NotImplementedError
