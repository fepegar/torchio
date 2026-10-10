"""Patch samplers for training and inference."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset

from ..types import TypeThreeInts
from .patch import PatchLocation
from .subject import Subject

if TYPE_CHECKING:
    from ..transforms.spatial._padding import PaddingMode


class PatchSampler:
    """Base class for patch samplers.

    Args:
        patch_size: Spatial size of each patch. A single `int`
            is broadcast to all three axes.
    """

    def __init__(self, patch_size: int | TypeThreeInts) -> None:
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size: TypeThreeInts = patch_size

    def __call__(
        self,
        subject: Subject,
        num_patches: int | None = None,
    ) -> Iterator[Subject]:
        """Sample patches from a subject.

        Override in subclasses for custom sampling strategies.

        Args:
            subject: Subject to extract patches from.
            num_patches: Number of patches to yield. If `None`,
                yields indefinitely (for random samplers) or yields
                all grid positions (for GridSampler).
        """
        msg = f"{type(self).__name__} must implement __call__"
        raise NotImplementedError(msg)

    def _extract_patch(
        self,
        subject: Subject,
        location: PatchLocation,
    ) -> Subject:
        """Extract a patch from a subject at the given location."""
        si, sj, sk = location.to_slices()
        kwargs: dict[str, Any] = {}
        for name, image in subject.images.items():
            kwargs[name] = image[:, si, sj, sk]
        for name in subject.metadata:
            kwargs[name] = subject.metadata[name]
        kwargs["patch_location"] = location
        return Subject(**kwargs)


class GridSampler(PatchSampler, Dataset):
    """Extract patches on a regular grid for dense inference.

    A map-style `Dataset` with known length and random access.
    Pass directly to a `DataLoader` for batched inference.
    Typically used with
    [`PatchAggregator`][torchio.data.PatchAggregator].

    Args:
        subject: Subject to extract patches from.
        patch_size: Spatial size of each patch.
        patch_overlap: Overlap between adjacent patches. Must be even.
            A single `int` is broadcast to all axes.
        padding_mode: If not `None`, pad the volume by
            `overlap // 2` on each side before sampling.
        fill: Fill value when `padding_mode='constant'`.

    Examples:
        >>> sampler = tio.GridSampler(subject, patch_size=64, patch_overlap=8)
        >>> loader = DataLoader(sampler, batch_size=4)
        >>> aggregator = tio.PatchAggregator(subject.spatial_shape, overlap_mode="hann")
        >>> for batch in loader:
        ...     outputs = model(batch.t1.data)
        ...     aggregator.add_batch(outputs, batch.patch_location)
        >>> volume = aggregator.get_output()
    """

    def __init__(
        self,
        subject: Subject,
        patch_size: int | TypeThreeInts,
        patch_overlap: int | TypeThreeInts = 0,
        padding_mode: PaddingMode | None = None,
        fill: float = 0,
    ) -> None:
        super().__init__(patch_size)
        if isinstance(patch_overlap, int):
            patch_overlap = (patch_overlap, patch_overlap, patch_overlap)
        self.patch_overlap: TypeThreeInts = patch_overlap
        self.padding_mode = padding_mode
        self.fill = fill
        self.subject = self._maybe_pad(subject)
        self.locations = self._compute_locations(self.subject.spatial_shape)

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, index: int) -> Subject:
        return self._extract_patch(self.subject, self.locations[index])

    def _maybe_pad(self, subject: Subject) -> Subject:
        if self.padding_mode is None:
            return subject
        from ..transforms.spatial.pad import Pad

        border = tuple(v // 2 for v in self.patch_overlap)
        padding = (
            border[0],
            border[0],
            border[1],
            border[1],
            border[2],
            border[2],
        )
        pad = Pad(
            padding=padding,
            padding_mode=self.padding_mode,
            fill=self.fill,
            copy=False,
        )
        return pad(subject)

    def _compute_locations(
        self,
        spatial_shape: TypeThreeInts,
    ) -> list[PatchLocation]:
        """Compute grid locations covering the volume."""
        locations: list[PatchLocation] = []
        indices_per_axis: list[list[int]] = []
        for dim in range(3):
            size = spatial_shape[dim]
            patch = self.patch_size[dim]
            overlap = self.patch_overlap[dim]
            step = max(patch - overlap, 1)
            indices = list(range(0, size - patch + 1, step))
            if not indices or indices[-1] != size - patch:
                indices.append(max(size - patch, 0))
            indices_per_axis.append(indices)

        for i in indices_per_axis[0]:
            for j in indices_per_axis[1]:
                for k in indices_per_axis[2]:
                    locations.append(
                        PatchLocation(
                            index=(i, j, k),
                            size=self.patch_size,
                        ),
                    )
        return locations


class UniformSampler(PatchSampler, IterableDataset):
    """Random patches with uniform spatial probability.

    An `IterableDataset` for training. Also callable for use with
    [`Queue`][torchio.data.Queue].

    Args:
        subject: Subject to sample patches from (for Dataset use).
        patch_size: Spatial size of each patch.
        num_patches: Number of patches per epoch. If `None`,
            yields indefinitely.

    Examples:
        >>> sampler = tio.UniformSampler(subject, patch_size=64, num_patches=100)
        >>> loader = DataLoader(sampler, batch_size=8)
    """

    def __init__(
        self,
        subject: Subject,
        patch_size: int | TypeThreeInts,
        num_patches: int | None = None,
    ) -> None:
        super().__init__(patch_size)
        self.subject = subject
        self.num_patches = num_patches

    def __call__(
        self,
        subject: Subject,
        num_patches: int | None = None,
    ) -> Iterator[Subject]:
        """Sample random patches from a given subject."""
        limit = num_patches or self.num_patches
        count = 0
        while limit is None or count < limit:
            index = self._random_index(subject.spatial_shape)
            loc = PatchLocation(index=index, size=self.patch_size)
            yield self._extract_patch(subject, loc)
            count += 1

    def __iter__(self) -> Iterator[Subject]:
        return self(self.subject, self.num_patches)

    def _random_index(
        self,
        spatial_shape: TypeThreeInts,
    ) -> TypeThreeInts:
        def _rand(d: int) -> int:
            hi = max(spatial_shape[d] - self.patch_size[d], 0) + 1
            return int(torch.randint(0, hi, (1,)).item())

        return (_rand(0), _rand(1), _rand(2))


class WeightedSampler(PatchSampler, IterableDataset):
    """Random patches weighted by a probability map.

    An `IterableDataset` for training with spatial priors.

    Args:
        subject: Subject to sample patches from.
        patch_size: Spatial size of each patch.
        probability_map: Name of the image in the subject to use
            as sampling weights.
        num_patches: Number of patches per epoch. If `None`,
            yields indefinitely.
    """

    def __init__(
        self,
        subject: Subject,
        patch_size: int | TypeThreeInts,
        probability_map: str,
        num_patches: int | None = None,
    ) -> None:
        super().__init__(patch_size)
        self.subject = subject
        self.probability_map = probability_map
        self.num_patches = num_patches

    def __call__(
        self,
        subject: Subject,
        num_patches: int | None = None,
    ) -> Iterator[Subject]:
        """Sample weighted patches from a given subject."""
        prob_data = self._build_probability_map_for(subject)
        flat = prob_data.flatten()
        if flat.sum() == 0:
            msg = f"Probability map '{self.probability_map}' is all zeros"
            raise RuntimeError(msg)

        limit = num_patches or self.num_patches
        count = 0
        while limit is None or count < limit:
            idx_flat = torch.multinomial(flat, 1).item()
            center = tuple(
                int(x) for x in np.unravel_index(int(idx_flat), prob_data.shape)
            )
            index = _center_to_corner(center, subject.spatial_shape, self.patch_size)
            loc = PatchLocation(index=index, size=self.patch_size)
            yield self._extract_patch(subject, loc)
            count += 1

    def __iter__(self) -> Iterator[Subject]:
        return self(self.subject, self.num_patches)

    def _build_probability_map_for(self, subject: Subject) -> Tensor:
        prob_image = subject.images[self.probability_map]
        prob_data = prob_image.data[0].float()
        return _mask_borders(prob_data, subject.spatial_shape, self.patch_size)

    def _build_probability_map(self) -> Tensor:
        return self._build_probability_map_for(self.subject)


class LabelSampler(WeightedSampler):
    """Random patches centered on labeled voxels.

    An `IterableDataset` for training with class imbalance.

    Args:
        subject: Subject to sample patches from.
        patch_size: Spatial size of each patch.
        label_name: Name of the label image in the subject.
        label_probabilities: Dict mapping label values to sampling
            weights. If `None`, all non-zero labels have equal
            weight.
        num_patches: Number of patches per epoch.
    """

    def __init__(
        self,
        subject: Subject,
        patch_size: int | TypeThreeInts,
        label_name: str,
        label_probabilities: dict[int, float] | None = None,
        num_patches: int | None = None,
    ) -> None:
        super().__init__(
            subject,
            patch_size,
            probability_map=label_name,
            num_patches=num_patches,
        )
        self.label_name = label_name
        self.label_probabilities = label_probabilities

    def _build_probability_map_for(self, subject: Subject) -> Tensor:
        label_image = subject.images[self.label_name]
        label_data = label_image.data[0]

        if self.label_probabilities is not None:
            prob = torch.zeros_like(label_data, dtype=torch.float32)
            for label, weight in self.label_probabilities.items():
                prob[label_data == label] = weight
        else:
            prob = (label_data > 0).float()

        return _mask_borders(prob, subject.spatial_shape, self.patch_size)

    def _build_probability_map(self) -> Tensor:
        return self._build_probability_map_for(self.subject)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_borders(
    prob: Tensor,
    spatial_shape: TypeThreeInts,
    patch_size: TypeThreeInts,
) -> Tensor:
    """Zero probability near borders where a patch center can't be placed."""
    prob = prob.clone()
    for d in range(3):
        half = patch_size[d] // 2
        if half > 0:
            slices_lo: list[slice] = [slice(None)] * 3
            slices_lo[d] = slice(0, half)
            prob[tuple(slices_lo)] = 0
        # An even-sized patch extends one voxel less beyond its center.
        tail = spatial_shape[d] - (patch_size[d] - 1) // 2
        if tail < spatial_shape[d]:
            slices_hi: list[slice] = [slice(None)] * 3
            slices_hi[d] = slice(tail, None)
            prob[tuple(slices_hi)] = 0
    return prob


def _center_to_corner(
    center: tuple[int, ...],
    spatial_shape: TypeThreeInts,
    patch_size: TypeThreeInts,
) -> TypeThreeInts:
    """Convert a center voxel to the patch corner index."""
    result: list[int] = []
    for d in range(3):
        half = patch_size[d] // 2
        corner = max(0, center[d] - half)
        corner = min(corner, spatial_shape[d] - patch_size[d])
        result.append(corner)
    return (result[0], result[1], result[2])
