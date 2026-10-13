"""LabelsToImage: generate a synthetic image from a label map."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from einops import rearrange
from torch import Tensor

from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from ...data.image import ScalarImage
from ..parameter_range import to_range
from ..transform import Transform


class LabelsToImage(Transform):
    r"""Generate a synthetic image from a label map.

    For each label, Gaussian-distributed tissue is created with a
    sampled mean and standard deviation, weighted by the label mask.
    The per-label contributions are summed to produce the output
    image.

    This is the building block behind
    [SynthSeg](https://github.com/BBillot/SynthSeg)-style synthesis.
    For best results, compose with
    [`Blur`][torchio.Blur] and
    [`BiasField`][torchio.BiasField].

    The generated image is added to the subject under the key given
    by *image_key*.  Existing images are **not** modified.

    Only [`LabelMap`][torchio.LabelMap] images are used as input.

    Args:
        label_key: Name of the label map to use.  If `None`, the
            first `LabelMap` found is used.
        image_key: Name for the generated `ScalarImage`.
        mean: Per-label mean ranges.  If `None`, each label gets a
            mean sampled from *default_mean*.
        std: Per-label std ranges.  If `None`, each label gets a
            std sampled from *default_std*.
        default_mean: Fallback range for label means.
        default_std: Fallback range for label stds.
        ignore_background: If `True`, label 0 is left as zero.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.LabelsToImage(label_key="seg")
        >>> transform = tio.LabelsToImage(
        ...     label_key="seg",
        ...     mean=[(0.8, 1.0), (0.3, 0.5)],
        ...     std=[(0.01, 0.05), (0.02, 0.08)],
        ... )
    """

    def __init__(
        self,
        label_key: str | None = None,
        *,
        image_key: str = "image_from_labels",
        mean: Sequence[float | tuple[float, float]] | None = None,
        std: Sequence[float | tuple[float, float]] | None = None,
        default_mean: float | tuple[float, float] = (0.1, 0.9),
        default_std: float | tuple[float, float] = (0.01, 0.1),
        ignore_background: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.label_key = label_key
        self.image_key = image_key
        self.mean_ranges = [to_range(m) for m in mean] if mean is not None else None
        self.std_ranges = [to_range(s) for s in std] if std is not None else None
        self.default_mean = to_range(default_mean)
        self.default_std = to_range(default_std)
        self.ignore_background = ignore_background

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample per-label mean and std values (per element when batched)."""
        label_batch = self._find_label_batch(batch)
        # Discover unique labels from the first sample.
        unique = sorted(int(v) for v in label_batch.data[0].unique().tolist())
        n = self._resolve_n(batch)
        if n is None:
            means, stds = self._sample_label_values(unique)
            return {"means": means, "stds": stds}
        means_list: list[dict[int, float]] = []
        stds_list: list[dict[int, float]] = []
        for _ in range(n):
            means, stds = self._sample_label_values(unique)
            means_list.append(means)
            stds_list.append(stds)
        params = {"means": means_list, "stds": stds_list}
        self._tag_batched(params, batch, n, None, ["means", "stds"])
        return params

    @property
    def supports_per_instance_params(self) -> bool:
        return True

    def _sample_label_values(
        self,
        unique: list[int],
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Sample one mean and std per label.

        Args:
            unique: Sorted list of unique label values.

        Returns:
            A `(means, stds)` pair of per-label dictionaries.
        """
        means: dict[int, float] = {}
        stds: dict[int, float] = {}
        for idx, label in enumerate(unique):
            if self.ignore_background and label == 0:
                means[label] = 0.0
                stds[label] = 0.0
                continue
            if self.mean_ranges is not None and idx < len(self.mean_ranges):
                means[label] = self.mean_ranges[idx].sample_1d()
            else:
                means[label] = self.default_mean.sample_1d()
            if self.std_ranges is not None and idx < len(self.std_ranges):
                stds[label] = self.std_ranges[idx].sample_1d()
            else:
                stds[label] = abs(self.default_std.sample_1d())
        return means, stds

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Generate a synthetic image and add it to the batch."""
        label_batch = self._find_label_batch(batch)
        if self._is_per_instance_params(params):
            generated = _generate_per_element(
                label_batch.data,
                params["means"],
                params["stds"],
            )
        else:
            generated = _generate_from_labels(
                label_batch.data,
                params["means"],
                params["stds"],
            )
        # Create a new image batch entry.
        from ...data.batch import ImagesBatch

        new_batch = ImagesBatch(
            data=generated,
            affines=label_batch.affines,
            image_class=ScalarImage,
        )
        batch.images[self.image_key] = new_batch
        return batch

    def _find_label_batch(self, batch: SubjectsBatch) -> Any:
        """Find the label map batch to use."""
        if self.label_key is not None:
            if self.label_key not in batch.images:
                msg = (
                    f"Label key '{self.label_key}' not found. "
                    f"Available: {list(batch.images.keys())}"
                )
                raise KeyError(msg)
            return batch.images[self.label_key]
        # Auto-detect first LabelMap.
        for _name, img_batch in self._get_images(batch).items():
            if issubclass(img_batch._image_class, LabelMap):
                return img_batch
        msg = "No LabelMap found in the subject"
        raise KeyError(msg)


def _generate_per_element(
    label_data: Tensor,
    means_per_element: list[dict[int, float]],
    stds_per_element: list[dict[int, float]],
) -> Tensor:
    """Generate synthetic tissue with per-element label statistics.

    Args:
        label_data: `(B, C, I, J, K)` label tensor.
        means_per_element: One per-label mean dict per batch element.
        stds_per_element: One per-label std dict per batch element.

    Returns:
        `(B, 1, I, J, K)` synthetic image tensor.
    """
    b = label_data.shape[0]
    spatial = label_data.shape[2:]
    result = torch.zeros(b, 1, *spatial, device=label_data.device)

    for label_val in _label_values_from(means_per_element):
        means = _broadcast_values(
            means_per_element,
            label_val,
            result,
        )
        stds = _broadcast_values(
            stds_per_element,
            label_val,
            result,
        )
        if _is_all_zero(means) and _is_all_zero(stds):
            continue
        mask = (label_data[:, 0:1] == label_val).to(dtype=result.dtype)
        tissue = torch.randn_like(result) * stds + means
        result += tissue * mask

    return result


def _label_values_from(values_per_element: list[dict[int, float]]) -> list[int]:
    """Get sorted label values represented by per-element dictionaries.

    Args:
        values_per_element: One value dictionary per batch element.

    Returns:
        Sorted union of labels in the dictionaries.
    """
    return sorted(set().union(*(values.keys() for values in values_per_element)))


def _broadcast_values(
    values_per_element: list[dict[int, float]],
    label_val: int,
    reference: Tensor,
) -> Tensor:
    """Convert per-element values for one label to a broadcastable tensor.

    Args:
        values_per_element: One value dictionary per batch element.
        label_val: Label whose values are needed.
        reference: Tensor defining device and dtype.

    Returns:
        Tensor of shape `(B, 1, 1, 1, 1)`, on the same device and dtype as
            `reference`, broadcastable over `(B, 1, I, J, K)`.
    """
    values = [values.get(label_val, 0.0) for values in values_per_element]
    tensor = torch.as_tensor(
        values,
        device=reference.device,
        dtype=reference.dtype,
    )
    return rearrange(tensor, "b -> b 1 1 1 1")


def _is_all_zero(values: Tensor) -> bool:
    """Return whether all tensor entries are zero."""
    return torch.count_nonzero(values).item() == 0


def _generate_from_labels(
    label_data: Tensor,
    means: dict[int, float],
    stds: dict[int, float],
) -> Tensor:
    """Generate Gaussian tissue for each label.

    Args:
        label_data: `(B, C, I, J, K)` label tensor.
        means: Per-label mean values.
        stds: Per-label std values.

    Returns:
        `(B, 1, I, J, K)` synthetic image tensor.
    """
    b = label_data.shape[0]
    spatial = label_data.shape[2:]
    result = torch.zeros(b, 1, *spatial, device=label_data.device)

    for label_val, mean in means.items():
        std = stds.get(label_val, 0.0)
        if mean == 0.0 and std == 0.0:
            continue
        mask = (label_data[:, 0:1] == label_val).float()
        tissue = torch.randn_like(result) * std + mean
        result += tissue * mask

    return result
