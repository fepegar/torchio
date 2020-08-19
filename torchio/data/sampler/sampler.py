import copy
from typing import Tuple, Optional, Generator

import numpy as np

from ... import TypePatchSize, TypeTripletInt
from ...data.subject import Subject
from ...utils import to_tuple


class PatchSampler:
    r"""Base class for TorchIO samplers.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`h \times w \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.
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
            sample: Subject,
            index_ini: TypeTripletInt,
            ) -> Subject:
        index_ini = np.array(index_ini)
        index_fin = index_ini + self.patch_size
        cropped_sample = sample.crop(index_ini, index_fin)
        cropped_sample['index_ini'] = index_ini.astype(int)
        return cropped_sample


class RandomSampler(PatchSampler):
    r"""Base class for TorchIO samplers.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`h \times w \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.
    """
    def __call__(
            self,
            sample: Subject,
            num_patches: Optional[int] = None,
            ) -> Generator[Subject, None, None]:
        raise NotImplementedError

    def get_probability_map(self, sample: Subject):
        raise NotImplementedError
