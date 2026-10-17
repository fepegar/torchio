"""CopyAffine: copy the affine matrix from a reference image."""

from __future__ import annotations

import copy
from typing import Any

from ...data.batch import SubjectsBatch
from ..transform import SpatialTransform


class CopyAffine(SpatialTransform):
    """Copy the affine matrix from one image to all others.

    This is useful when slight numerical differences between affine
    matrices cause downstream errors (e.g., in
    [`Resample`][torchio.Resample]).  NIfTI stores affines in
    single precision, so saving and reloading can introduce
    rounding errors.

    Args:
        target: Name of the image whose affine will be copied to
            all other images in the subject.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.CopyAffine(target="t1")
    """

    def __init__(self, target: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.target = target

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """No random parameters."""
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Copy the reference affine to all other images."""
        if self.target not in batch.images:
            msg = (
                f"Reference image '{self.target}' not found. "
                f"Available: {list(batch.images.keys())}"
            )
            raise KeyError(msg)
        ref_affines = batch.images[self.target].affines
        for name, img_batch in self._get_images(batch).items():
            if name == self.target:
                continue
            for i, affine in enumerate(img_batch.affines):
                affine._matrix = copy.deepcopy(ref_affines[i]._matrix)
        return batch
