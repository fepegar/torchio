import warnings
from typing import Tuple

import torch
import numpy as np

from ...constants import CHANNELS_DIMENSION
from ..sampler import GridSampler


class GridAggregator:
    r"""Aggregate patches for dense inference.

    This class is typically used to build a volume made of patches after
    inference of batches extracted by a :class:`~torchio.data.GridSampler`.

    Args:
        sampler: Instance of :class:`~torchio.data.GridSampler` used to
            extract the patches.
        overlap_mode: If ``'crop'``, the overlapping predictions will be
            cropped. If ``'average'``, the predictions in the overlapping areas
            will be averaged with equal weights. See the
            `grid aggregator tests`_ for a raw visualization of both modes.

    .. _grid aggregator tests: https://github.com/fepegar/torchio/blob/main/tests/data/inference/test_aggregator.py

    .. note:: Adapted from NiftyNet. See `this NiftyNet tutorial
        <https://niftynet.readthedocs.io/en/dev/window_sizes.html>`_ for more
        information about patch-based sampling.
    """  # noqa: E501
    def __init__(self, sampler: GridSampler, overlap_mode: str = 'crop'):
        subject = sampler.subject
        self.volume_padded = sampler.padding_mode is not None
        self.spatial_shape = subject.spatial_shape
        self._output_tensor = None
        self.patch_overlap = sampler.patch_overlap
        self._parse_overlap_mode(overlap_mode)
        self.overlap_mode = overlap_mode
        self._avgmask_tensor = None

    @staticmethod
    def _parse_overlap_mode(overlap_mode):
        if overlap_mode not in ('crop', 'average'):
            message = (
                'Overlap mode must be "crop" or "average" but '
                f' "{overlap_mode}" was passed'
            )
            raise ValueError(message)

    def _crop_patch(
            self,
            patch: torch.Tensor,
            location: np.ndarray,
            overlap: np.ndarray,
            ) -> Tuple[torch.Tensor, np.ndarray]:
        half_overlap = overlap // 2  # overlap is always even in grid sampler
        index_ini, index_fin = location[:3], location[3:]

        # If the patch is not at the border, we crop half the overlap
        crop_ini = half_overlap.copy()
        crop_fin = half_overlap.copy()

        # If the volume has been padded, we don't need to worry about cropping
        if self.volume_padded:
            pass
        else:
            crop_ini *= index_ini > 0
            crop_fin *= index_fin != self.spatial_shape

        # Update the location of the patch in the volume
        new_index_ini = index_ini + crop_ini
        new_index_fin = index_fin - crop_fin
        new_location = np.hstack((new_index_ini, new_index_fin))

        patch_size = patch.shape[-3:]
        i_ini, j_ini, k_ini = crop_ini
        i_fin, j_fin, k_fin = patch_size - crop_fin
        cropped_patch = patch[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]
        return cropped_patch, new_location

    def _initialize_output_tensor(self, batch: torch.Tensor) -> None:
        if self._output_tensor is not None:
            return
        num_channels = batch.shape[CHANNELS_DIMENSION]
        self._output_tensor = torch.zeros(
            num_channels,
            *self.spatial_shape,
            dtype=batch.dtype,
        )

    def _initialize_avgmask_tensor(self, batch: torch.Tensor) -> None:
        if self._avgmask_tensor is not None:
            return
        num_channels = batch.shape[CHANNELS_DIMENSION]
        self._avgmask_tensor = torch.zeros(
            num_channels,
            *self.spatial_shape,
            dtype=batch.dtype,
        )

    def add_batch(
            self,
            batch_tensor: torch.Tensor,
            locations: torch.Tensor,
            ) -> None:
        """Add batch processed by a CNN to the output prediction volume.

        Args:
            batch_tensor: 5D tensor, typically the output of a convolutional
                neural network, e.g. ``batch['image'][torchio.DATA]``.
            locations: 2D tensor with shape :math:`(B, 6)` representing the
                patch indices in the original image. They are typically
                extracted using ``batch[torchio.LOCATION]``.
        """
        batch = batch_tensor.cpu()
        locations = locations.cpu().numpy()
        patch_sizes = locations[:, 3:] - locations[:, :3]
        # There should be only one patch size
        assert len(np.unique(patch_sizes, axis=0)) == 1
        input_spatial_shape = tuple(batch.shape[-3:])
        target_spatial_shape = tuple(patch_sizes[0])
        if input_spatial_shape != target_spatial_shape:
            message = (
                f'The shape of the input batch, {input_spatial_shape},'
                ' does not match the shape of the target location,'
                f' which is {target_spatial_shape}'
            )
            raise RuntimeError(message)
        self._initialize_output_tensor(batch)
        if self.overlap_mode == 'crop':
            for patch, location in zip(batch, locations):
                cropped_patch, new_location = self._crop_patch(
                    patch,
                    location,
                    self.patch_overlap,
                )
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = new_location
                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin] = cropped_patch
        elif self.overlap_mode == 'average':
            self._initialize_avgmask_tensor(batch)
            for patch, location in zip(batch, locations):
                i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
                self._output_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin] += patch
                self._avgmask_tensor[
                    :,
                    i_ini:i_fin,
                    j_ini:j_fin,
                    k_ini:k_fin] += 1

    def get_output_tensor(self) -> torch.Tensor:
        """Get the aggregated volume after dense inference."""
        if self._output_tensor.dtype == torch.int64:
            message = (
                'Medical image frameworks such as ITK do not support int64.'
                ' Casting to int32...'
            )
            warnings.warn(message, RuntimeWarning)
            self._output_tensor = self._output_tensor.type(torch.int32)
        if self.overlap_mode == 'average':
            # true_divide is used instead of / in case the PyTorch version is
            # old and one the operands is int:
            # https://github.com/fepegar/torchio/issues/526
            output = torch.true_divide(
                self._output_tensor, self._avgmask_tensor)
        else:
            output = self._output_tensor
        if self.volume_padded:
            from ...transforms import Crop
            border = self.patch_overlap // 2
            cropping = border.repeat(2)
            crop = Crop(cropping)
            return crop(output)
        else:
            return output
