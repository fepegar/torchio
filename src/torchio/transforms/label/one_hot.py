"""OneHot: one-hot encode label maps."""

from __future__ import annotations

from typing import Any

import torch.nn.functional as functional

from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from ..transform import Transform


class OneHot(Transform):
    r"""One-hot encode label maps.

    Each label map with $K$ classes (including background) is converted
    from shape $(1, I, J, K)$ to $(K, I, J, K)$, where channel $k$
    is 1 where the label equals $k$ and 0 elsewhere.

    Only [`LabelMap`][torchio.LabelMap] images are affected.
    [`ScalarImage`][torchio.ScalarImage] instances are left unchanged.

    The inverse takes the argmax across channels, restoring the
    original single-channel label map.

    Args:
        num_classes: Total number of classes. `-1` (default) infers
            from the data as `max_label + 1`.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.OneHot()
        >>> transform = tio.OneHot(num_classes=5)
        >>> # Invert back to single-channel
        >>> restored = transformed.apply_inverse_transform()
    """

    def __init__(
        self,
        *,
        num_classes: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """No random parameters."""
        return {"num_classes": self.num_classes}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """One-hot encode each label map in the batch."""
        num_classes = params["num_classes"]
        for _name, img_batch in self._get_images(batch).items():
            if not issubclass(img_batch._image_class, LabelMap):
                continue
            # (B, 1, I, J, K) -> (B, num_classes, I, J, K)
            data = img_batch.data.long()
            flat = data[:, 0]  # (B, I, J, K)
            encoded = functional.one_hot(flat, num_classes=num_classes)
            # (B, I, J, K, num_classes) -> (B, num_classes, I, J, K)
            img_batch.data = encoded.permute(0, 4, 1, 2, 3).float()
        return batch

    @property
    def invertible(self) -> bool:
        """Whether this transform can be inverted."""
        return True

    def inverse(self, params: dict[str, Any]) -> _OneHotInverse:
        """Invert by taking argmax."""
        return _OneHotInverse(copy=False)


class _OneHotInverse(Transform):
    """Inverse of OneHot: argmax back to single-channel labels."""

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        for _name, img_batch in self._get_images(batch).items():
            if not issubclass(img_batch._image_class, LabelMap):
                continue
            if img_batch.data.shape[1] > 1:
                img_batch.data = img_batch.data.argmax(dim=1, keepdim=True).float()
        return batch
