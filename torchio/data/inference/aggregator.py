from typing import Tuple
import torch
import numpy as np
from ...utils import to_tuple
from ...torchio import TypeData, TypeTuple
from ..subject import Subject


class GridAggregator:
    r"""Aggregate patches for dense inference.

    This class is typically used to build a volume made of batches after
    inference of patches extracted by a :py:class:`~torchio.data.GridSampler`.

    Args:
        sample: Instance of :py:class:`~torchio.data.subject.Subject`
            from which patches will be extracted (probably using a
            :py:class:`~torchio.data.GridSampler`).
        patch_overlap: Tuple of integers :math:`(d_o, h_o, w_o)` specifying the
            overlap between patches. If a single number
            :math:`n` is provided, :math:`d_o = h_o = w_o = n`.
        out_channels: Number of channels in the output tensor.

    .. note:: Adapted from NiftyNet. See `this NiftyNet tutorial
        <https://niftynet.readthedocs.io/en/dev/window_sizes.html>`_ for more
        information.
    """
    def __init__(
            self,
            sample: Subject,
            patch_overlap: TypeTuple,
            out_channels: int = 1,
            ):
        self._output_tensor = torch.zeros(out_channels, *sample.shape)
        self.patch_overlap = to_tuple(patch_overlap, length=3)

    @staticmethod
    def _crop_batch(
            patches: torch.Tensor,
            location: np.ndarray,
            border: Tuple[int, int, int],
            ) -> Tuple[TypeData, np.ndarray]:
        location = location.astype(np.int)
        batch_shape = patches.shape
        spatial_shape = batch_shape[2:]  # ignore batch and channels dim
        num_dimensions = 3
        for idx in range(num_dimensions):
            location[:, idx] = location[:, idx] + border[idx]
            location[:, idx + 3] = location[:, idx + 3] - border[idx]
        cropped_shape = np.max(location[:, 3:6] - location[:, 0:3], axis=0)
        diff = spatial_shape - cropped_shape
        left = np.floor(diff / 2).astype(np.int)
        i_ini, j_ini, k_ini = left
        i_fin, j_fin, k_fin = left + cropped_shape
        batch = patches[
            :,  # batch dimension
            :,  # channels dimension
            i_ini:i_fin,
            j_ini:j_fin,
            k_ini:k_fin,
        ]
        return batch, location

    def _ensure_output_dtype(self, tensor: torch.Tensor) -> None:
        """Make sure the output tensor type is the same as the input patches."""
        if self._output_tensor.dtype != tensor.dtype:
            self._output_tensor = self._output_tensor.type(tensor.dtype)

    def add_batch(self, patches: torch.Tensor, locations: TypeData) -> None:
        patches = patches.cpu()
        self._ensure_output_dtype(patches)
        location_init = np.copy(locations)
        init_ones = np.ones_like(patches)
        patches, _ = self._crop_batch(
            patches, location_init, self.patch_overlap)
        location_init = np.copy(locations)
        _, locations = self._crop_batch(
            init_ones, location_init, self.patch_overlap)
        for patch, location in zip(patches, locations):
            i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
            channels = len(patch)
            for channel in range(channels):
                self._output_tensor[channel, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = patch[channel]

    def get_output_tensor(self) -> torch.Tensor:
        return self._output_tensor
