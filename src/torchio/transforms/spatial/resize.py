"""Resize: resample to a target shape (not spacing)."""

from __future__ import annotations

from typing import Any

import torch.nn.functional as functional

from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from ..transform import SpatialTransform


class Resize(SpatialTransform):
    r"""Resize images to a target spatial shape.

    The field of view is preserved; voxel spacing is adjusted to fit
    the new shape.

    Warning:
        In most medical image applications, this transform should
        **not** be used as it scales anisotropically.  Prefer
        [`Resample`][torchio.Resample] (change spacing) combined with
        [`CropOrPad`][torchio.CropOrPad] (change shape) instead.

    Args:
        target_shape: Target spatial shape $(I, J, K)$.  A single
            integer $N$ means $(N, N, N)$.
        image_interpolation: `"linear"` (default) for intensity
            images.
        label_interpolation: `"nearest"` (default) for label maps.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Resize(128)
        >>> transform = tio.Resize((256, 256, 64))
    """

    def __init__(
        self,
        target_shape: int | tuple[int, int, int],
        *,
        image_interpolation: str = "linear",
        label_interpolation: str = "nearest",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(target_shape, int):
            target_shape = (target_shape, target_shape, target_shape)
        self.target_shape = target_shape
        self.image_interpolation = image_interpolation
        self.label_interpolation = label_interpolation

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """No random parameters."""
        return {"target_shape": self.target_shape}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Resize each image to the target shape."""
        target = list(params["target_shape"])
        for _name, img_batch in self._get_images(batch).items():
            is_label = issubclass(img_batch._image_class, LabelMap)
            mode = self.label_interpolation if is_label else self.image_interpolation
            torch_mode = "nearest" if mode == "nearest" else "trilinear"
            old_shape = img_batch.data.shape[2:]
            img_batch.data = functional.interpolate(
                img_batch.data.float(),
                size=target,
                mode=torch_mode,
                align_corners=None if torch_mode == "nearest" else True,
            ).to(img_batch.data.dtype)
            # Update affines: spacing changes to fit new shape in same FOV.
            for affine in img_batch.affines:
                for axis in range(3):
                    factor = old_shape[axis] / target[axis]
                    affine._matrix[:3, axis] *= factor
        return batch
