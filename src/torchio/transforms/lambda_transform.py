"""Lambda: apply a user-defined callable as a transform."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor

from ..data.batch import SubjectsBatch
from ..data.image import LabelMap
from ..data.image import ScalarImage
from .transform import Transform


class Lambda(Transform):
    """Apply a user-defined function as a transform.

    The callable receives and returns a 4D tensor `(C, I, J, K)`.
    Use *types_to_apply* to restrict which image types are affected.

    Args:
        function: Callable that receives and returns a 4D
            [`torch.Tensor`][torch.Tensor].
        types_to_apply: Which image types the function applies to.
            `"scalar"` for [`ScalarImage`][torchio.ScalarImage] only,
            `"label"` for [`LabelMap`][torchio.LabelMap] only,
            `None` for all images.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> invert = tio.Lambda(lambda x: -x, types_to_apply="scalar")
        >>> double = tio.Lambda(lambda x: 2 * x)
        >>> threshold = tio.Lambda(lambda x: (x > 0.5).float(), types_to_apply="label")
    """

    def __init__(
        self,
        function: Callable[[Tensor], Tensor],
        types_to_apply: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not callable(function):
            msg = f"function must be callable, got {type(function).__name__}"
            raise TypeError(msg)
        self.function = function
        self.types_to_apply = types_to_apply

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """No random parameters."""
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Apply the callable to each matching image."""
        for _name, img_batch in self._get_images(batch).items():
            if not self._should_apply(img_batch._image_class):
                continue
            for i in range(img_batch.batch_size):
                img_batch.data[i] = self.function(img_batch.data[i])
        return batch

    def _should_apply(self, image_class: type) -> bool:
        """Check whether this image type should be transformed."""
        if self.types_to_apply is None:
            return True
        if self.types_to_apply == "scalar":
            return issubclass(image_class, ScalarImage)
        if self.types_to_apply == "label":
            return issubclass(image_class, LabelMap)
        return True
