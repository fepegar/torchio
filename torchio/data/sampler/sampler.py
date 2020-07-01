import copy
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
