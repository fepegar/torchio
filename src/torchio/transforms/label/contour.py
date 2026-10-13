"""Contour: extract label boundaries."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from ..transform import Transform


class Contour(Transform):
    r"""Keep only the boundary voxels of each label.

    A voxel is on the boundary if any of its 6-connected neighbors
    has a different value.  The result is a binary mask of the
    contours.

    Only [`LabelMap`][torchio.LabelMap] images are affected.

    Args:
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Contour()
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """No random parameters."""
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Replace each label map with its boundary voxels."""
        for _name, img_batch in self._get_images(batch).items():
            if not issubclass(img_batch._image_class, LabelMap):
                continue
            img_batch.data = _extract_contour(img_batch.data)
        return batch


def _extract_contour(data: torch.Tensor) -> torch.Tensor:
    """Extract boundaries using 3D erosion via max-pooling.

    A voxel is interior if all its 6-connected neighbors have the
    same value as itself.  We detect this by comparing the original
    with a morphological erosion (min of neighbors).

    Args:
        data: `(B, C, I, J, K)` label tensor.

    Returns:
        Binary `(B, C, I, J, K)` tensor: 1 on boundaries, 0 inside.
    """
    # Pad with -1 so boundary voxels at the edge are detected.
    padded = functional.pad(data.float(), [1] * 6, mode="constant", value=-1)
    # Min-pool with kernel 3 gives the morphological erosion.
    eroded = -functional.max_pool3d(-padded, kernel_size=3, stride=1, padding=0)
    # A voxel is on the contour if eroded != original (neighbor differs).
    contour = (eroded != data.float()).float()
    return contour
