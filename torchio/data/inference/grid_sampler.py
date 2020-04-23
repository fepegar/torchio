import copy
from typing import Tuple

import torch
import numpy as np
from torch.utils.data import Dataset

from ...utils import to_tuple
from ...torchio import LOCATION, TypeTuple, DATA
from ..subject import Subject


class GridSampler(Dataset):
    r"""Extract patches across a whole volume.

    Grid samplers are useful to perform inference using all patches from a
    volume. It is often used with a
    :py:class:`~torchio.data.GridAggregator`.

    Args:
        sample: Instance of:py:class:`~torchio.data.subject.Subject`
            from which patches will be extracted.
        patch_size: Tuple of integers :math:`(d, h, w)` to generate patches
            of size :math:`d \times h \times w`.
            If a single number :math:`n` is provided,
            :math:`d = h = w = n`.
        patch_overlap: Tuple of integers :math:`(d_o, h_o, w_o)` specifying the
            overlap between patches for dense inference. If a single number
            :math:`n` is provided, :math:`d_o = h_o = w_o = n`.

    .. note:: Adapted from NiftyNet. See `this NiftyNet tutorial
        <https://niftynet.readthedocs.io/en/dev/window_sizes.html>`_ for more
        information.
    """
    def __init__(
            self,
            sample: Subject,
            patch_size: TypeTuple,
            patch_overlap: TypeTuple,
            ):
        self.sample = sample
        patch_size = to_tuple(patch_size, length=3)
        patch_overlap = to_tuple(patch_overlap, length=3)
        self.locations = self._grid_spatial_coordinates(
            self.sample.shape,
            patch_size,
            patch_overlap,
        )

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        # Assume 3D
        location = self.locations[index]
        index_ini = location[:3]
        index_fin = location[3:]
        cropped_sample = self.extract_patch(self.sample, index_ini, index_fin)
        cropped_sample[LOCATION] = location
        return cropped_sample

    def extract_patch(
            self,
            sample: Subject,
            index_ini: Tuple[int, int, int],
            index_fin: Tuple[int, int, int],
            ) -> Subject:
        cropped_sample = self.copy_and_crop(
            sample,
            index_ini,
            index_fin,
        )
        return cropped_sample

    @staticmethod
    def copy_and_crop(
            sample: Subject,
            index_ini: np.ndarray,
            index_fin: np.ndarray,
            ) -> dict:
        cropped_sample = {}
        iterable = sample.get_images_dict(intensity_only=False).items()
        for image_name, image in iterable:
            cropped_sample[image_name] = copy.deepcopy(image)
            sample_image_dict = image
            cropped_image_dict = cropped_sample[image_name]
            cropped_image_dict[DATA] = crop(
                sample_image_dict[DATA], index_ini, index_fin)
        # torch doesn't like uint16
        cropped_sample['index_ini'] = index_ini.astype(int)
        return cropped_sample

    @staticmethod
    def _enumerate_step_points(
            starting: int,
            ending: int,
            win_size: int,
            step_size: int,
            ) -> np.ndarray:
        starting = max(int(starting), 0)
        ending = max(int(ending), 0)
        win_size = max(int(win_size), 1)
        step_size = max(int(step_size), 1)
        if starting > ending:
            starting, ending = ending, starting
        sampling_point_set = []
        while (starting + win_size) <= ending:
            sampling_point_set.append(starting)
            starting = starting + step_size
        additional_last_point = ending - win_size
        sampling_point_set.append(max(additional_last_point, 0))
        sampling_point_set = np.unique(sampling_point_set).flatten()
        if len(sampling_point_set) == 2:
            sampling_point_set = np.append(
                sampling_point_set, np.round(np.mean(sampling_point_set)))
        _, uniq_idx = np.unique(sampling_point_set, return_index=True)
        return sampling_point_set[np.sort(uniq_idx)]

    @staticmethod
    def _grid_spatial_coordinates(
            volume_shape: Tuple[int, int, int],
            window_shape: Tuple[int, int, int],
            border: Tuple[int, int, int],
            ) -> np.ndarray:
        num_dims = len(volume_shape)
        grid_size = [
            max(win_size - 2 * border, 0)
            for (win_size, border)
            in zip(window_shape, border)
        ]
        steps_along_each_dim = [
            GridSampler._enumerate_step_points(
                starting=0,
                ending=volume_shape[i],
                win_size=window_shape[i],
                step_size=grid_size[i],
            )
            for i in range(num_dims)
        ]
        starting_coords = np.asanyarray(np.meshgrid(*steps_along_each_dim))
        starting_coords = starting_coords.reshape((num_dims, -1)).T
        n_locations = starting_coords.shape[0]
        # prepare the output coordinates matrix
        spatial_coords = np.zeros((n_locations, num_dims * 2), dtype=np.int32)
        spatial_coords[:, :num_dims] = starting_coords
        for idx in range(num_dims):
            spatial_coords[:, num_dims + idx] = (
                starting_coords[:, idx]
                + window_shape[idx]
            )
        max_coordinates = np.max(spatial_coords, axis=0)[num_dims:]
        assert np.all(max_coordinates <= volume_shape[:num_dims]), \
            "window size greater than the spatial coordinates {} : {}".format(
                max_coordinates, volume_shape)
        return spatial_coords


def crop(
        image: torch.Tensor,
        index_ini: np.ndarray,
        index_fin: np.ndarray,
        ) -> torch.Tensor:
    i_ini, j_ini, k_ini = index_ini
    i_fin, j_fin, k_fin = index_fin
    return image[..., i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
