"""Swap: randomly swap patches within an image for self-supervised learning."""

from __future__ import annotations

import warnings
from typing import Any

import torch
from einops import rearrange
from torch import Tensor

from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from ..parameter_range import to_nonneg_range
from ..transform import IntensityTransform
from ..transform import Transform

Origin = tuple[int, int, int]
SwapLocation = tuple[Origin, Origin]
PatchIndices = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


class Swap(IntensityTransform):
    r"""Randomly swap patches within an image.

    This is typically used in
    [context restoration for self-supervised learning](https://www.sciencedirect.com/science/article/pii/S1361841518304699).
    Pairs of same-sized patches are selected at random and their
    contents are exchanged.

    Warning:
        This transform is intended for **self-supervised** or
        **unsupervised** workflows.  Because the spatial content is
        rearranged, aligned label maps become inconsistent with the
        swapped image.  A warning is emitted if `LabelMap` images
        are present in the subject.

    Args:
        patch_size: Spatial size of the patches to swap.  A single
            integer $n$ means $(n, n, n)$.
        num_iterations: Number of patch pairs to swap.  A 2-tuple
            $(a, b)$ samples $n \sim \mathcal{U}(a, b)$.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Swap(patch_size=15, num_iterations=100)
    """

    def __init__(
        self,
        *,
        patch_size: int | tuple[int, int, int] = 15,
        num_iterations: int | tuple[int, int] = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.num_iterations = to_nonneg_range(num_iterations)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample swap locations (per element when batched)."""
        # Warn if label maps are present. Use the base Transform._get_images
        # (include/exclude only) rather than IntensityTransform's override,
        # which filters to ScalarImage and would never see a LabelMap.
        for _name, img_batch in Transform._get_images(self, batch).items():
            if issubclass(img_batch._image_class, LabelMap):
                warnings.warn(
                    "Swap is applied to a subject containing LabelMap "
                    "images. The spatial rearrangement will make labels "
                    "inconsistent with the swapped image. This transform "
                    "is intended for self-supervised learning.",
                    stacklevel=2,
                )
                break

        any_img = next(iter(batch.images.values()))
        spatial_shape = any_img.data.shape[2:]  # (I, J, K)

        n = self._resolve_n(batch)
        if n is None:
            iterations = max(1, round(self.num_iterations.sample_1d()))
            locations = _sample_swap_locations(
                spatial_shape,
                self.patch_size,
                iterations,
            )
            return {"locations": locations}

        keep = self._keep_mask(batch, n)
        locations_list: list[Any] = []
        for index in range(n):
            if keep is not None and not keep[index]:
                locations_list.append([])
                continue
            iterations = max(1, round(self.num_iterations.sample_1d()))
            locations_list.append(
                _sample_swap_locations(spatial_shape, self.patch_size, iterations)
            )
        params = {"locations": locations_list}
        self._tag_batched(params, batch, n, keep, ["locations"])
        return params

    @property
    def supports_per_instance_params(self) -> bool:
        return True

    @property
    def supports_per_instance_p(self) -> bool:
        return True

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Swap patches in each selected image."""
        per_instance = self._is_per_instance_params(params)
        for _name, img_batch in self._get_images(batch).items():
            if per_instance:
                img_batch.data = _apply_swaps_per_instance(
                    img_batch.data,
                    params["locations"],
                    self.patch_size,
                )
            else:
                img_batch.data = _apply_swaps(
                    img_batch.data,
                    params["locations"],
                    self.patch_size,
                )
        return batch


def _sample_swap_locations(
    spatial_shape: tuple[int, ...],
    patch_size: tuple[int, int, int],
    num_iterations: int,
) -> list[SwapLocation]:
    """Sample pairs of non-overlapping patch origins.

    Args:
        spatial_shape: `(I, J, K)` spatial dimensions.
        patch_size: `(pi, pj, pk)` patch dimensions.
        num_iterations: Number of pairs to sample.

    Returns:
        List of `(origin_a, origin_b)` tuples.
    """
    locations: list[SwapLocation] = []
    max_ini = [s - p for s, p in zip(spatial_shape, patch_size, strict=True)]
    if any(m < 0 for m in max_ini):
        msg = (
            f"Patch size {patch_size} cannot be larger than "
            f"spatial shape {tuple(spatial_shape)}"
        )
        raise ValueError(msg)

    for _ in range(num_iterations):
        first = _random_origin(max_ini)
        # Resample second until non-overlapping with first.
        for _ in range(100):
            second = _random_origin(max_ini)
            if not _patches_overlap(first, second, patch_size):
                break
        locations.append((first, second))

    return locations


def _random_origin(
    max_ini: list[int],
) -> Origin:
    """Sample a random patch origin."""
    coords = []
    for m in max_ini:
        if m == 0:
            coords.append(0)
        else:
            coords.append(int(torch.randint(m + 1, (1,)).item()))
    return (coords[0], coords[1], coords[2])


def _patches_overlap(
    a: Origin,
    b: Origin,
    patch_size: tuple[int, int, int],
) -> bool:
    """Check whether two axis-aligned patches overlap."""
    for ai, bi, p in zip(a, b, patch_size, strict=True):
        if ai + p <= bi or bi + p <= ai:
            return False
    return True


def _apply_swaps(
    data: Tensor,
    locations: list[SwapLocation],
    patch_size: tuple[int, int, int],
) -> Tensor:
    """Swap patch pairs in a 5D tensor.

    Args:
        data: `(B, C, I, J, K)` tensor.
        locations: List of `(origin_a, origin_b)` pairs.
        patch_size: `(pi, pj, pk)` patch dimensions.

    Returns:
        Tensor with patches swapped.
    """
    result = data.clone()
    pi, pj, pk = patch_size

    for (ai, aj, ak), (bi, bj, bk) in locations:
        patch_a = result[:, :, ai : ai + pi, aj : aj + pj, ak : ak + pk].clone()
        patch_b = result[:, :, bi : bi + pi, bj : bj + pj, bk : bk + pk].clone()
        result[:, :, ai : ai + pi, aj : aj + pj, ak : ak + pk] = patch_b
        result[:, :, bi : bi + pi, bj : bj + pj, bk : bk + pk] = patch_a

    return result


def _apply_swaps_per_instance(
    data: Tensor,
    locations: list[list[SwapLocation]],
    patch_size: tuple[int, int, int],
) -> Tensor:
    """Swap per-element patch pairs in a 5D tensor.

    Args:
        data: `(B, C, I, J, K)` tensor.
        locations: One list of `(origin_a, origin_b)` pairs per batch element.
        patch_size: `(pi, pj, pk)` patch dimensions.

    Returns:
        Tensor with each element's patches swapped.
    """
    result = data.clone()
    num_swaps = max(
        (len(element_locations) for element_locations in locations), default=0
    )
    if num_swaps == 0:
        return result

    origins_a, origins_b = _get_batched_origins(
        locations,
        num_swaps,
        data.device,
    )
    patch_indices = _make_patch_indices(data, patch_size)
    for swap_index in range(num_swaps):
        _swap_batched_patches(
            result,
            origins_a[:, swap_index],
            origins_b[:, swap_index],
            patch_indices,
        )

    return result


def _get_batched_origins(
    locations: list[list[SwapLocation]],
    num_swaps: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Build origin tensors for batched indexed swapping.

    Args:
        locations: One list of `(origin_a, origin_b)` pairs per batch element.
        num_swaps: Number of sequential swap steps to encode.
        device: Device on which the index tensors are created.

    Returns:
        Two tensors of shape `(B, num_swaps, 3)` for the first and second patch
            origins.
    """
    batch_size = len(locations)
    origins_a = torch.zeros(batch_size, num_swaps, 3, dtype=torch.long, device=device)
    origins_b = torch.zeros_like(origins_a)
    for batch_index, element_locations in enumerate(locations):
        for swap_index, (origin_a, origin_b) in enumerate(element_locations):
            origins_a[batch_index, swap_index] = torch.as_tensor(
                origin_a,
                dtype=torch.long,
                device=device,
            )
            origins_b[batch_index, swap_index] = torch.as_tensor(
                origin_b,
                dtype=torch.long,
                device=device,
            )
    return origins_a, origins_b


def _make_patch_indices(
    data: Tensor,
    patch_size: tuple[int, int, int],
) -> PatchIndices:
    """Create shared batch, channel, and patch-offset index tensors.

    Args:
        data: `(B, C, I, J, K)` tensor.
        patch_size: `(pi, pj, pk)` patch dimensions.

    Returns:
        Index tensors that broadcast to `(B, C, pi, pj, pk)`.
    """
    batch_size, channels = data.shape[:2]
    pi, pj, pk = patch_size
    device = data.device
    batch_index = rearrange(
        torch.arange(batch_size, device=device),
        "b -> b 1 1 1 1",
    )
    channel_index = rearrange(
        torch.arange(channels, device=device),
        "c -> 1 c 1 1 1",
    )
    i_offsets = rearrange(torch.arange(pi, device=device), "i -> 1 1 i 1 1")
    j_offsets = rearrange(torch.arange(pj, device=device), "j -> 1 1 1 j 1")
    k_offsets = rearrange(torch.arange(pk, device=device), "k -> 1 1 1 1 k")
    return batch_index, channel_index, i_offsets, j_offsets, k_offsets


def _swap_batched_patches(
    data: Tensor,
    origins_a: Tensor,
    origins_b: Tensor,
    patch_indices: PatchIndices,
) -> None:
    """Swap one patch pair per batch element using batched indexing.

    Args:
        data: `(B, C, I, J, K)` tensor to update in place.
        origins_a: Tensor of shape `(B, 3)` with first patch origins.
        origins_b: Tensor of shape `(B, 3)` with second patch origins.
        patch_indices: Broadcastable batch, channel, and offset indices.
    """
    indices_a = _get_patch_indices(origins_a, patch_indices)
    indices_b = _get_patch_indices(origins_b, patch_indices)
    patch_a = data[indices_a].clone()
    patch_b = data[indices_b].clone()
    data[indices_a] = patch_b
    data[indices_b] = patch_a


def _get_patch_indices(
    origins: Tensor,
    patch_indices: PatchIndices,
) -> PatchIndices:
    """Build full tensor indices for per-element patch origins.

    Args:
        origins: Tensor of shape `(B, 3)` with per-element patch origins.
        patch_indices: Broadcastable batch, channel, and offset indices.

    Returns:
        Index tensors that select one patch per batch element.
    """
    batch_index, channel_index, i_offsets, j_offsets, k_offsets = patch_indices
    i_index = rearrange(origins[:, 0], "b -> b 1 1 1 1") + i_offsets
    j_index = rearrange(origins[:, 1], "b -> b 1 1 1 1") + j_offsets
    k_index = rearrange(origins[:, 2], "b -> b 1 1 1 1") + k_offsets
    return batch_index, channel_index, i_index, j_index, k_index
