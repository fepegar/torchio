"""RemapLabels: reassign label values in a label map."""

from __future__ import annotations

from typing import Any

from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from ..transform import Transform


class RemapLabels(Transform):
    r"""Reassign label values in label maps.

    Each key in the *remapping* dict is replaced by its value.
    Labels not mentioned in the dict are left unchanged.

    Only [`LabelMap`][torchio.LabelMap] images are affected.

    Args:
        remapping: Dictionary mapping old labels to new labels.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> # Merge labels 2 and 3 into label 1
        >>> transform = tio.RemapLabels({2: 1, 3: 1})
        >>> # Swap labels 1 and 2
        >>> transform = tio.RemapLabels({1: 2, 2: 1})
    """

    def __init__(
        self,
        remapping: dict[int, int],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.remapping = remapping

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """No random parameters."""
        return {"remapping": self.remapping}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Remap labels in each label map."""
        remapping = params["remapping"]
        for _name, img_batch in self._get_images(batch).items():
            if not issubclass(img_batch._image_class, LabelMap):
                continue
            data = img_batch.data.clone()
            for old, new in remapping.items():
                data[img_batch.data == old] = new
            img_batch.data = data
        return batch

    @property
    def invertible(self) -> bool:
        """Whether this transform can be inverted."""
        return True

    def inverse(self, params: dict[str, Any]) -> RemapLabels:
        """Invert by swapping keys and values."""
        remapping = params["remapping"]
        inverse_remapping = {v: k for k, v in remapping.items()}
        return RemapLabels(remapping=inverse_remapping, copy=False)
