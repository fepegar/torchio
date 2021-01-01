from typing import Optional, Tuple, Generator

import torch
import numpy as np

from ...typing import TypePatchSize
from ..image import Image
from ..subject import Subject
from .sampler import RandomSampler


class WeightedSampler(RandomSampler):
    r"""Randomly extract patches from a volume given a probability map.

    The probability of sampling a patch centered on a specific voxel is the
    value of that voxel in the probability map. The probabilities need not be
    normalized. For example, voxels can have values 0, 1 and 5. Voxels with
    value 0 will never be at the center of a patch. Voxels with value 5 will
    have 5 times more chance of being at the center of a patch that voxels
    with a value of 1.

    Args:
        patch_size: See :class:`~torchio.data.PatchSampler`.
        probability_map: Name of the image in the input subject that will be
            used as a sampling probability map.

    Raises:
        RuntimeError: If the probability map is empty.

    Example:
        >>> import torchio as tio
        >>> subject = tio.Subject(
        ...     t1=tio.ScalarImage('t1_mri.nii.gz'),
        ...     sampling_map=tio.Image('sampling.nii.gz', type=tio.SAMPLING_MAP),
        ... )
        >>> patch_size = 64
        >>> sampler = tio.data.WeightedSampler(patch_size, 'sampling_map')
        >>> for patch in sampler(subject):
        ...     print(patch['index_ini'])

    .. note:: The index of the center of a patch with even size :math:`s` is
        arbitrarily set to :math:`s/2`. This is an implementation detail that
        will typically not make any difference in practice.

    .. note:: Values of the probability map near the border will be set to 0 as
        the center of the patch cannot be at the border (unless the patch has
        size 1 or 2 along that axis).

    """  # noqa: E501
    def __init__(
            self,
            patch_size: TypePatchSize,
            probability_map: str,
            ):
        super().__init__(patch_size)
        self.probability_map_name = probability_map
        self.cdf = None

    def __call__(
            self,
            subject: Subject,
            num_patches: Optional[int] = None,
            ) -> Generator[Subject, None, None]:
        subject.check_consistent_space()
        if np.any(self.patch_size > subject.spatial_shape):
            message = (
                f'Patch size {tuple(self.patch_size)} cannot be'
                f' larger than image size {tuple(subject.spatial_shape)}'
            )
            raise RuntimeError(message)
        probability_map = self.get_probability_map(subject)
        probability_map = self.process_probability_map(
            probability_map, subject)
        cdf = self.get_cumulative_distribution_function(probability_map)

        patches_left = num_patches if num_patches is not None else True
        while patches_left:
            yield self.extract_patch(subject, probability_map, cdf)
            if num_patches is not None:
                patches_left -= 1

    def get_probability_map_image(self, subject: Subject) -> Image:
        if self.probability_map_name in subject:
            return subject[self.probability_map_name]
        else:
            message = (
                f'Image "{self.probability_map_name}"'
                f' not found in subject: {subject}'
            )
            raise KeyError(message)

    def get_probability_map(self, subject: Subject) -> torch.Tensor:
        data = self.get_probability_map_image(subject).data
        if torch.any(data < 0):
            message = (
                'Negative values found'
                f' in probability map "{self.probability_map_name}"'
            )
            raise ValueError(message)
        return data

    def process_probability_map(
            self,
            probability_map: torch.Tensor,
            subject: Subject,
            ) -> np.ndarray:
        # Using float32 can create cdf with maximum very far from 1, e.g. 0.92!
        data = probability_map[0].numpy().astype(np.float64)
        assert data.ndim == 3
        self.clear_probability_borders(data, self.patch_size)
        total = data.sum()
        if total == 0:
            half_patch_size = tuple(n // 2 for n in self.patch_size)
            message = (
                'Empty probability map found:'
                f' {self.get_probability_map_image(subject).path}'
                '\nVoxels with positive probability might be near the image'
                ' border.\nIf you suspect that this is the case, try adding a'
                ' padding transform\nwith half the patch size:'
                f' torchio.Pad({half_patch_size})'
            )
            raise RuntimeError(message)
        data /= total  # normalize probabilities
        return data

    @staticmethod
    def clear_probability_borders(
            probability_map: np.ndarray,
            patch_size: TypePatchSize,
            ) -> None:
        # Set probability to 0 on voxels that wouldn't possibly be sampled
        # given the current patch size
        # We will arbitrarily define the center of an array with even length
        # using the // Python operator
        # For example, the center of an array (3, 4) will be on (1, 2)
        #
        #   Patch         center
        #  . . . .        . . . .
        #  . . . .   ->   . . x .
        #  . . . .        . . . .
        #
        #
        #    Prob. map      After preprocessing
        #
        #  x x x x x x x       . . . . . . .
        #  x x x x x x x       . . x x x x .
        #  x x x x x x x  -->  . . x x x x .
        #  x x x x x x x  -->  . . x x x x .
        #  x x x x x x x       . . x x x x .
        #  x x x x x x x       . . . . . . .
        #
        # The dots represent removed probabilities, x mark possible locations
        crop_ini = patch_size // 2
        crop_fin = (patch_size - 1) // 2
        crop_i, crop_j, crop_k = crop_ini
        probability_map[:crop_i, :, :] = 0
        probability_map[:, :crop_j, :] = 0
        probability_map[:, :, :crop_k] = 0

        # The call tolist() is very important. Using np.uint16 as negative
        # index will not work because e.g. -np.uint16(2) == 65534
        crop_i, crop_j, crop_k = crop_fin.tolist()
        if crop_i:
            probability_map[-crop_i:, :, :] = 0
        if crop_j:
            probability_map[:, -crop_j:, :] = 0
        if crop_k:
            probability_map[:, :, -crop_k:] = 0

    @staticmethod
    def get_cumulative_distribution_function(
            probability_map: np.ndarray,
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the cumulative distribution function of a probability map."""
        flat_map = probability_map.flatten()
        flat_map_normalized = flat_map / flat_map.sum()
        cdf = np.cumsum(flat_map_normalized)
        return cdf

    def extract_patch(
            self,
            subject: Subject,
            probability_map: np.ndarray,
            cdf: np.ndarray
            ) -> Subject:
        index_ini = self.get_random_index_ini(probability_map, cdf)
        cropped_subject = self.crop(subject, index_ini, self.patch_size)
        cropped_subject['index_ini'] = index_ini.astype(int)
        return cropped_subject

    def get_random_index_ini(
            self,
            probability_map: np.ndarray,
            cdf: np.ndarray
            ) -> np.ndarray:
        center = self.sample_probability_map(probability_map, cdf)
        assert np.all(center >= 0)
        # See self.clear_probability_borders
        index_ini = center - self.patch_size // 2
        assert np.all(index_ini >= 0)
        return index_ini

    @classmethod
    def sample_probability_map(
            cls,
            probability_map: np.ndarray,
            cdf: np.ndarray
            ) -> np.ndarray:
        """Inverse transform sampling.

        Example:
            >>> probability_map = np.array(
            ...    ((0,0,1,1,5,2,1,1,0),
            ...     (2,2,2,2,2,2,2,2,2)))
            >>> probability_map
            array([[0, 0, 1, 1, 5, 2, 1, 1, 0],
                   [2, 2, 2, 2, 2, 2, 2, 2, 2]])
            >>> histogram = np.zeros_like(probability_map)
            >>> for _ in range(100000):
            ...     histogram[WeightedSampler.sample_probability_map(probability_map, cdf)] += 1  # doctest:+SKIP
            ...
            >>> histogram  # doctest:+SKIP
            array([[    0,     0,  3479,  3478, 17121,  7023,  3355,  3378,     0],
                   [ 6808,  6804,  6942,  6809,  6946,  6988,  7002,  6826,  7041]])

        """  # noqa: E501
        # Get first value larger than random number
        random_number = torch.rand(1).item()
        # If probability map is float32, cdf.max() can be far from 1, e.g. 0.92
        if random_number > cdf.max():
            cdf_index = -1
        else:  # proceed as usual
            cdf_index = np.searchsorted(cdf, random_number)

        random_location_index = cdf_index
        center = np.unravel_index(
            random_location_index,
            probability_map.shape
        )

        i, j, k = center
        probability = probability_map[i, j, k]
        assert probability > 0

        center = np.array(center).astype(int)
        return center
