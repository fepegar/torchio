"""RemoveLabels: set specified label values to background."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from ..transform import Transform


class RemoveLabels(Transform):
    """Set specified label values to a background value.

    This is a convenience wrapper that builds a remapping dict
    internally.  For more control, use
    [`RemapLabels`][torchio.RemapLabels] directly.

    Only [`LabelMap`][torchio.LabelMap] images are affected.

    Args:
        labels: Label values to remove.
        background_label: Value to assign to removed labels.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.RemoveLabels([3, 4, 5])
        >>> transform = tio.RemoveLabels([2], background_label=0)
    """

    def __init__(
        self,
        labels: Sequence[int],
        *,
        background_label: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.labels = list(labels)
        self.background_label = background_label

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """No random parameters."""
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Set specified labels to the background value."""
        for _name, img_batch in self._get_images(batch).items():
            if not issubclass(img_batch._image_class, LabelMap):
                continue
            data = img_batch.data.clone()
            for label in self.labels:
                data[img_batch.data == label] = self.background_label
            img_batch.data = data
        return batch
