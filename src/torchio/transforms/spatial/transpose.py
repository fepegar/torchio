"""Transpose: swap the first and last spatial axes."""

from __future__ import annotations

from typing import Any

from ...data.batch import SubjectsBatch
from ..transform import SpatialTransform


class Transpose(SpatialTransform):
    r"""Swap the first and last spatial dimensions.

    Transforms an image of shape $(C, I, J, K)$ into $(C, K, J, I)$.
    The affine matrix is updated to reflect the reordering so that
    world coordinates remain consistent.

    This is the v2 equivalent of v1's `Transpose`, which reversed
    the orientation string.  The transform is its own inverse.

    Args:
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Transpose()
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
        """Swap first and last spatial axes for all images."""
        for _name, img_batch in self._get_images(batch).items():
            # Swap axes 2 (I) and 4 (K) in the (B, C, I, J, K) tensor.
            img_batch.data = img_batch.data.permute(0, 1, 4, 3, 2).contiguous()
            # Update affines: swap columns 0 and 2 (I↔K).
            for affine in img_batch.affines:
                m = affine._matrix.clone()
                affine._matrix[:, 0] = m[:, 2]
                affine._matrix[:, 2] = m[:, 0]
        return batch

    @property
    def invertible(self) -> bool:
        """Transpose is its own inverse."""
        return True

    def inverse(self, params: dict[str, Any]) -> Transpose:
        """Transposing twice is identity."""
        return Transpose(copy=False)
