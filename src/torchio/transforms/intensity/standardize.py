"""Standardize: subtract mean and divide by standard deviation."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from torch import Tensor

from ...data.batch import ImagesBatch
from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from ..transform import IntensityTransform


class Standardize(IntensityTransform):
    r"""Subtract mean and divide by standard deviation (z-score).

    $$v_{\text{out}} = \frac{v - \mu}{\sigma}$$

    The statistics $\mu$ and $\sigma$ are computed from the (optionally
    masked) voxels and applied to the entire image.

    Args:
        masking_method: Which voxels to include when computing the
            mean and standard deviation. `None` uses all voxels.
            A `str` is interpreted as a key to a
            [`LabelMap`][torchio.LabelMap] in the subject.
            A callable receives the image tensor and returns a boolean
            mask.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Standardize()
        >>> # Use only brain voxels for statistics
        >>> transform = tio.Standardize(masking_method="brain")
        >>> # Use voxels above mean
        >>> transform = tio.Standardize(masking_method=lambda x: x > x.mean())
    """

    def __init__(
        self,
        *,
        masking_method: str | Callable[[Tensor], Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.masking_method = masking_method

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Compute mean and std for each image of each sample in the batch.

        Returns:
            Dict mapping image names to one `(mean, std)` pair per sample.
        """
        images = self._get_images(batch)
        stats: dict[str, list[tuple[float, float]]] = {}
        for name, img_batch in images.items():
            per_sample: list[tuple[float, float]] = []
            for index in range(len(img_batch.data)):
                mask = _get_mask(self.masking_method, img_batch, batch, index)
                tensor = img_batch.data[index]
                values = (
                    tensor[mask.expand_as(tensor)]
                    if mask is not None
                    else tensor.reshape(-1)
                )
                if values.numel() == 0:
                    warnings.warn(
                        f'Mask is empty for "{name}". Using all voxels.',
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    values = tensor.reshape(-1)
                mean = float(values.float().mean().item())
                std = float(values.float().std().item())
                per_sample.append((mean, std))
            stats[name] = per_sample
        return {"stats": stats}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Subtract mean and divide by std for each selected image."""
        stats = params["stats"]
        for name, img_batch in self._get_images(batch).items():
            if name not in stats:
                continue
            data = img_batch.data.float()
            for index, (mean, std) in enumerate(stats[name]):
                if std == 0:
                    msg = (
                        f"Standard deviation is zero for masked values in"
                        f' "{name}" (sample {index}). Cannot standardize.'
                    )
                    raise RuntimeError(msg)
                data[index] = (data[index] - mean) / std
            img_batch.data = data
        return batch

    @property
    def invertible(self) -> bool:
        """Whether this transform can be inverted."""
        return True

    def inverse(self, params: dict[str, Any]) -> _StandardizeInverse:
        """Build the inverse using the recorded mean and std."""
        return _StandardizeInverse(stats=params["stats"], copy=False)


class _StandardizeInverse(IntensityTransform):
    """Inverse of Standardize for history replay."""

    def __init__(
        self,
        *,
        stats: dict[str, list[tuple[float, float]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._stats = stats

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Return empty params; all state is in instance attributes."""
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Reverse the standardization: `data * std + mean`."""
        for name, img_batch in self._get_images(batch).items():
            if name not in self._stats:
                continue
            data = img_batch.data.float()
            for index, (mean, std) in enumerate(self._stats[name]):
                if std == 0:
                    continue
                data[index] = data[index] * std + mean
            img_batch.data = data
        return batch


def _get_mask(
    masking_method: str | Callable[[Tensor], Tensor] | None,
    img_batch: ImagesBatch,
    batch: SubjectsBatch,
    index: int = 0,
) -> Tensor | None:
    """Resolve a masking method to a boolean tensor or `None`.

    Args:
        masking_method: `None`, a string key, or a callable.
        img_batch: The image batch being processed.
        batch: The full subject batch (for string-key lookup).
        index: Index of the sample in the batch to build the mask for.

    Returns:
        Boolean mask tensor, or `None` for no masking.
    """
    if masking_method is None:
        return None
    if callable(masking_method) and not isinstance(masking_method, str):
        return masking_method(img_batch.data[index]).bool()
    if isinstance(masking_method, str):
        if masking_method not in batch.images:
            msg = (
                f'Masking method "{masking_method}" not found in batch images.'
                f" Available: {list(batch.images.keys())}"
            )
            raise KeyError(msg)
        mask_batch = batch.images[masking_method]
        if not issubclass(mask_batch._image_class, LabelMap):
            msg = f'Masking method "{masking_method}" must refer to a LabelMap.'
            raise TypeError(msg)
        return mask_batch.data[index].bool()
    msg = f"masking_method must be None, str, or callable, got {type(masking_method)}"
    raise TypeError(msg)


# Backwards-compatible alias.
ZNormalization = Standardize
