"""KeepLargestComponent: keep only the largest connected component per label."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import SimpleITK as sitk
import torch
from torch import Tensor

from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from ..transform import Transform


class KeepLargestComponent(Transform):
    r"""Keep only the largest connected component of each label.

    For each specified label value, connected-component analysis is
    performed and all but the largest component are removed (set to
    the background value).  This is useful for cleaning up noisy
    segmentation predictions.

    Only single-channel [`LabelMap`][torchio.LabelMap] images are
    affected.

    Args:
        labels: Label values to filter.  `None` means all non-zero
            labels found in the data.
        background_label: Value used for removed components.
        fully_connected: If `True`, use 26-connectivity (voxels
            sharing a corner are connected).  If `False`, use
            6-connectivity (face-connected only).
        **kwargs: See [`Transform`][torchio.Transform].

    Raises:
        RuntimeError: If a label map has more than one channel.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.KeepLargestComponent()
        >>> transform = tio.KeepLargestComponent(labels=[1, 2])
    """

    def __init__(
        self,
        labels: Sequence[int] | None = None,
        *,
        background_label: int = 0,
        fully_connected: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.labels = list(labels) if labels is not None else None
        self.background_label = background_label
        self.fully_connected = fully_connected

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """No random parameters."""
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Keep only the largest connected component per label."""
        for _name, img_batch in self._get_images(batch).items():
            if not issubclass(img_batch._image_class, LabelMap):
                continue
            b, c = img_batch.data.shape[:2]
            if c != 1:
                msg = (
                    "KeepLargestComponent requires single-channel"
                    f" label maps, got {c} channels"
                )
                raise RuntimeError(msg)
            for i in range(b):
                img_batch.data[i, 0] = _keep_largest_per_label(
                    img_batch.data[i, 0],
                    labels=self.labels,
                    background_label=self.background_label,
                    fully_connected=self.fully_connected,
                )
        return batch


def _keep_largest_per_label(
    data: Tensor,
    *,
    labels: list[int] | None,
    background_label: int,
    fully_connected: bool,
) -> Tensor:
    """Keep the largest connected component for each label.

    Args:
        data: `(I, J, K)` label tensor.
        labels: Which labels to filter.  `None` means all
            non-zero labels.
        background_label: Value for removed voxels.
        fully_connected: Whether to use 26- or 6-connectivity.

    Returns:
        Filtered `(I, J, K)` tensor.
    """
    result = data.clone()
    if labels is None:
        unique = data.unique().tolist()
        labels = [int(v) for v in unique if int(v) != background_label]

    for label in labels:
        binary = (data == label).cpu().numpy().astype("uint8")
        if binary.sum() == 0:
            continue
        sitk_img = sitk.GetImageFromArray(binary)
        cc = sitk.ConnectedComponent(sitk_img, fully_connected)
        relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
        cc_array = sitk.GetArrayFromImage(relabeled)
        # cc_array label 1 is the largest component; remove others.
        mask = torch.from_numpy((cc_array >= 2).astype("uint8")).to(data.device)
        result[mask.bool()] = background_label

    return result
