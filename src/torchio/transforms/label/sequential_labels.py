"""SequentialLabels: renumber labels to consecutive integers."""

from __future__ import annotations

from typing import Any

import torch

from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from ..transform import Transform


class SequentialLabels(Transform):
    r"""Renumber labels in label maps to consecutive integers starting from 0.

    For example, if a label map has values `{0, 5, 10}`, this
    transform remaps them to `{0, 1, 2}`.

    Only [`LabelMap`][torchio.LabelMap] images are affected.

    Note:
        The background (label 0) is always mapped to 0. Even if there
        are no zeros in the input, zero will appear in the output.

    Args:
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.SequentialLabels()
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Compute the remapping from the first sample's labels."""
        remappings: dict[str, dict[int, int]] = {}
        for name, img_batch in self._get_images(batch).items():
            if not issubclass(img_batch._image_class, LabelMap):
                continue
            unique = sorted(int(v) for v in img_batch.data[0].unique().tolist())
            remappings[name] = {old: new for new, old in enumerate(unique)}
        return {"remappings": remappings}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Apply sequential renumbering."""
        remappings = params["remappings"]
        for name, img_batch in self._get_images(batch).items():
            if name not in remappings:
                continue
            remapping = remappings[name]
            data = torch.zeros_like(img_batch.data)
            for old, new in remapping.items():
                data[img_batch.data == old] = new
            img_batch.data = data
        return batch

    @property
    def invertible(self) -> bool:
        """Whether this transform can be inverted."""
        return True

    def inverse(self, params: dict[str, Any]) -> _SequentialLabelsInverse:
        """Invert by restoring original label values."""
        return _SequentialLabelsInverse(
            remappings=params["remappings"],
            copy=False,
        )


class _SequentialLabelsInverse(Transform):
    """Inverse of SequentialLabels."""

    def __init__(
        self,
        *,
        remappings: dict[str, dict[int, int]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._remappings = remappings

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        for name, img_batch in self._get_images(batch).items():
            if name not in self._remappings:
                continue
            inverse = {v: k for k, v in self._remappings[name].items()}
            data = torch.zeros_like(img_batch.data)
            for old, new in inverse.items():
                data[img_batch.data == old] = new
            img_batch.data = data
        return batch
