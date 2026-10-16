"""Patch aggregator for dense inference."""

from __future__ import annotations

import torch
from torch import Tensor

from ..types import TypeThreeInts
from .patch import PatchLocation


class PatchAggregator:
    """Reassemble patches into a full volume.

    Handles overlapping patches with configurable blending modes.
    Supports outputs of different spatial sizes than the input
    patches (e.g., downsampled feature maps or embeddings).

    Args:
        spatial_shape: Output volume spatial shape `(I, J, K)`.
        overlap_mode: How to handle overlapping regions:
            `'crop'` keeps only non-overlapping centers (fast,
            best for argmax segmentation);
            `'average'` averages overlapping values (best for
            probabilistic outputs);
            `'hann'` uses Hann-window weighting (smoothest,
            best for continuous outputs).
        patch_overlap: The overlap used during sampling, needed
            for `'crop'` mode to compute how much to trim.
        output_shape: If the model output is spatially smaller
            than the input patch (e.g., due to strided
            convolutions), specify the output volume shape here.
            Patch locations will be scaled accordingly.

    Examples:
        >>> aggregator = tio.PatchAggregator(
        ...     spatial_shape=(256, 256, 176),
        ...     overlap_mode="hann",
        ... )
        >>> for batch in loader:
        ...     outputs = model(batch.t1.data)
        ...     aggregator.add_batch(outputs, locations)
        >>> volume = aggregator.get_output()
    """

    def __init__(
        self,
        spatial_shape: TypeThreeInts,
        overlap_mode: str = "crop",
        patch_overlap: int | TypeThreeInts = 0,
        output_shape: TypeThreeInts | None = None,
    ) -> None:
        _validate_overlap_mode(overlap_mode)
        self.input_spatial_shape = spatial_shape
        self.overlap_mode = overlap_mode

        if isinstance(patch_overlap, int):
            patch_overlap = (patch_overlap, patch_overlap, patch_overlap)
        self.patch_overlap: TypeThreeInts = patch_overlap

        if output_shape is not None:
            self.spatial_shape = output_shape
            self._scale = (
                output_shape[0] / spatial_shape[0],
                output_shape[1] / spatial_shape[1],
                output_shape[2] / spatial_shape[2],
            )
        else:
            self.spatial_shape = spatial_shape
            self._scale = (1.0, 1.0, 1.0)

        self._outputs: dict[str, Tensor] = {}
        self._counts: dict[str, Tensor] = {}
        self._hann_cache: dict[TypeThreeInts, Tensor] = {}

    def add_batch(
        self,
        batch: Tensor | dict[str, Tensor],
        locations: list[PatchLocation],
    ) -> None:
        """Add a batch of model outputs to the aggregation buffer.

        Args:
            batch: 5D tensor `(B, C, I, J, K)` or dict of such
                tensors keyed by name.
            locations: List of `PatchLocation` for each item in
                the batch.
        """
        tensors: dict[str, Tensor] = (
            {"__default__": batch} if isinstance(batch, Tensor) else batch
        )

        for key, tensor in tensors.items():
            tensor = tensor.cpu()
            for idx, loc in enumerate(locations):
                patch = tensor[idx]
                if self._scale != (1.0, 1.0, 1.0):
                    loc = loc.scaled(self._scale)
                self._add_patch(key, patch, loc)

    def get_output(self, key: str | None = None) -> Tensor:
        """Get the aggregated output volume.

        Args:
            key: Name of the output to retrieve. If `None` and
                only a single (unnamed) output was added, return it.

        Returns:
            The aggregated tensor with shape `(C, I, J, K)`.
        """
        resolve_key = key if key is not None else "__default__"
        if resolve_key not in self._outputs:
            available = [k for k in self._outputs if k != "__default__"]
            msg = f"No output for key {key!r}. Available: {available}"
            raise KeyError(msg)

        output = self._outputs[resolve_key]

        if self.overlap_mode in ("average", "hann"):
            counts = self._counts[resolve_key]
            # Hann weights can sum to less than one at covered voxels.
            counts = counts.masked_fill(counts == 0, 1)
            output = output / counts

        return output

    def _add_patch(
        self,
        key: str,
        patch: Tensor,
        location: PatchLocation,
    ) -> None:
        self._ensure_buffer(key, patch)
        match self.overlap_mode:
            case "crop":
                self._add_crop(key, patch, location)
            case "average":
                self._add_average(key, patch, location)
            case "hann":
                self._add_hann(key, patch, location)

    def _ensure_buffer(self, key: str, patch: Tensor) -> None:
        if key in self._outputs:
            return
        num_channels = patch.shape[0]
        self._outputs[key] = torch.zeros(
            num_channels,
            *self.spatial_shape,
            dtype=patch.dtype,
        )
        if self.overlap_mode in ("average", "hann"):
            self._counts[key] = torch.zeros(
                num_channels,
                *self.spatial_shape,
                dtype=patch.dtype,
            )

    def _add_crop(
        self,
        key: str,
        patch: Tensor,
        location: PatchLocation,
    ) -> None:
        """Place only the non-overlapping center of the patch."""
        scaled_overlap = (
            round(self.patch_overlap[0] * self._scale[0]),
            round(self.patch_overlap[1] * self._scale[1]),
            round(self.patch_overlap[2] * self._scale[2]),
        )
        half = [o // 2 for o in scaled_overlap]
        ini = list(location.index_ini)
        fin = list(location.index_fin)
        crop_ini = [0, 0, 0]
        crop_fin = list(location.size)

        for d in range(3):
            if ini[d] > 0:
                ini[d] += half[d]
                crop_ini[d] += half[d]
            if fin[d] < self.spatial_shape[d]:
                fin[d] -= half[d]
                crop_fin[d] -= half[d]

        cropped = patch[
            :,
            crop_ini[0] : crop_fin[0],
            crop_ini[1] : crop_fin[1],
            crop_ini[2] : crop_fin[2],
        ]
        self._outputs[key][
            :,
            ini[0] : fin[0],
            ini[1] : fin[1],
            ini[2] : fin[2],
        ] = cropped

    def _add_average(
        self,
        key: str,
        patch: Tensor,
        location: PatchLocation,
    ) -> None:
        si, sj, sk = location.to_slices()
        self._outputs[key][:, si, sj, sk] += patch
        self._counts[key][:, si, sj, sk] += 1

    def _add_hann(
        self,
        key: str,
        patch: Tensor,
        location: PatchLocation,
    ) -> None:
        patch_shape = (
            patch.shape[-3],
            patch.shape[-2],
            patch.shape[-1],
        )
        window = self._get_hann_window(patch_shape)
        si, sj, sk = location.to_slices()
        self._outputs[key][:, si, sj, sk] += patch * window
        self._counts[key][:, si, sj, sk] += window

    def _get_hann_window(self, patch_size: TypeThreeInts) -> Tensor:
        if patch_size in self._hann_cache:
            return self._hann_cache[patch_size]
        window = _build_hann_3d(patch_size)
        self._hann_cache[patch_size] = window
        return window


def _validate_overlap_mode(mode: str) -> None:
    valid = ("crop", "average", "hann")
    if mode not in valid:
        msg = f"overlap_mode must be one of {valid}, got {mode!r}"
        raise ValueError(msg)


def _build_hann_3d(patch_size: TypeThreeInts) -> Tensor:
    """Build a 3D Hann window for smooth patch blending."""
    window = torch.ones(1)
    for dim, size in enumerate(patch_size):
        shape = [1, 1, 1]
        shape[dim] = size
        w = torch.hann_window(size + 2, periodic=False)[1:-1]
        window = window * w.reshape(shape)
    return window
