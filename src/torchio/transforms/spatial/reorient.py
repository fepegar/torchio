"""Reorient transform: reorder voxel axes to a target orientation."""

from __future__ import annotations

from typing import Any

import nibabel as nib
import numpy as np
import torch
from nibabel import orientations
from torch import Tensor

from ...data.batch import SubjectsBatch
from ..transform import SpatialTransform


def _validate_orientation(orientation: str) -> str:
    """Validate and normalise a 3-letter orientation code."""
    if not isinstance(orientation, str) or len(orientation) != 3:
        msg = f'Orientation must be a 3-letter string, got "{orientation}"'
        raise ValueError(msg)

    orientation = orientation.upper()
    valid_codes = set("RLAPIS")
    if not all(c in valid_codes for c in orientation):
        msg = (
            "Orientation code must be composed of three distinct characters"
            f' in {valid_codes} but got "{orientation}"'
        )
        raise ValueError(msg)

    _check_axis_coverage(orientation)
    return orientation


def _check_axis_coverage(orientation: str) -> None:
    """Ensure the orientation code covers all three axis pairs."""
    pairs = [{"R", "L"}, {"A", "P"}, {"S", "I"}]
    codes = set(orientation)
    if not all(codes & pair for pair in pairs):
        msg = (
            "Orientation code must include one character for each axis"
            f' direction: R or L, A or P, and S or I, but got "{orientation}"'
        )
        raise ValueError(msg)


def _compute_reorientation(
    current_affine: np.ndarray,
    target_codes: str,
) -> np.ndarray:
    """Compute the ornt_transform from current affine to target codes.

    Returns:
        (3, 2) array where column 0 is the input axis index and
        column 1 is the flip direction (1 or -1).
    """
    current_ornt = orientations.io_orientation(current_affine)
    target_ornt = orientations.axcodes2ornt(tuple(target_codes))
    return orientations.ornt_transform(current_ornt, target_ornt)


def _apply_reorientation(
    data: Tensor,
    ornt: np.ndarray,
) -> Tensor:
    """Apply an orientation transform to a tensor.

    Mirrors nibabel's `apply_orientation`: flip axes first, then
    transpose.  Works on both 4D `(C, I, J, K)` and 5D
    `(B, C, I, J, K)` tensors.  Spatial axes are always the last 3.

    Args:
        data: Input tensor.
        ornt: `(3, 2)` orientation transform from nibabel.
    """
    n_leading = data.ndim - 3

    # Step 1: flip axes where direction is -1 (on original axis indices).
    for ax in range(3):
        if ornt[ax, 1] == -1:
            data = torch.flip(data, [ax + n_leading])

    # Step 2: permute using argsort of the source-axis column, matching
    # nibabel's `arr.transpose(np.argsort(ornt[:, 0]))`.
    perm = np.argsort(ornt[:, 0]).astype(int)
    leading = list(range(n_leading))
    spatial_perm = [int(p) + n_leading for p in perm]
    data = data.permute(leading + spatial_perm)

    return data.contiguous()


class Reorient(SpatialTransform):
    r"""Reorder voxel axes to match a target orientation.

    The voxels are permuted and/or flipped so that the image axes
    align with the specified anatomical directions. The affine matrix
    is updated to preserve the physical positions of the voxels.

    Common orientation codes:

    - `'RAS'`: Left→\ **R**\ ight,
      Posterior→\ **A**\ nterior, Inferior→\ **S**\ uperior.
    - `'LPS'`: Right→\ **L**\ eft,
      Anterior→\ **P**\ osterior, Inferior→\ **S**\ uperior.

    See the `NiBabel docs on image orientation
    <https://nipy.org/nibabel/image_orientation.html>`__ for details.

    Args:
        orientation: Target three-letter orientation code. Must
            contain one letter for each axis pair: R/L, A/P, S/I.
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Reorient()            # default: RAS
        >>> transform = tio.Reorient(orientation='LPS')
        >>> transform = tio.Reorient(orientation='SPL')
    """

    def __init__(
        self,
        orientation: str = "RAS",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.orientation = _validate_orientation(orientation)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        # Images in a subject may have different orientations, so the
        # reorientation is computed for each image separately.
        ornts: dict[str, list] = {}
        original_orientations: dict[str, str] = {}
        for name, img_batch in self._get_images(batch).items():
            target = self._get_target_orientation(name)
            if target is None:
                continue
            affine_np = img_batch.affines[0].numpy()
            original_orientations[name] = "".join(
                nib.orientations.aff2axcodes(affine_np),
            )
            ornts[name] = _compute_reorientation(affine_np, target).tolist()
        return {
            "ornt": ornts,
            "original_orientation": original_orientations,
        }

    def _get_target_orientation(self, name: str) -> str | None:
        """Return the orientation code that the image should end up in."""
        return self.orientation

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        for name, img_batch in self._get_images(batch).items():
            if name not in params["ornt"]:
                continue
            ornt = np.asarray(params["ornt"][name])

            # No-op when already in target orientation
            is_identity = np.array_equal(ornt[:, 0], [0, 1, 2]) and np.all(
                ornt[:, 1] == 1
            )
            if is_identity:
                continue

            original_shape = img_batch.data.shape[-3:]
            img_batch.data = _apply_reorientation(img_batch.data, ornt)

            for affine in img_batch.affines:
                inv_aff = orientations.inv_ornt_aff(ornt, original_shape)
                new_matrix = affine.numpy() @ inv_aff
                affine._matrix = torch.as_tensor(
                    new_matrix,
                    dtype=torch.float64,
                )

        return batch

    @property
    def invertible(self) -> bool:
        return True

    def inverse(self, params: dict[str, Any]) -> _ReorientInverse:
        """Inverse reorients each image back to its original orientation."""
        return _ReorientInverse(
            orientations=params["original_orientation"],
            copy=False,
        )


class _ReorientInverse(Reorient):
    """Inverse of [`Reorient`][torchio.Reorient] for history replay.

    Reorients each image back to the orientation it had before the
    forward transform was applied.
    """

    def __init__(
        self,
        *,
        orientations: dict[str, str],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._orientations = orientations

    def _get_target_orientation(self, name: str) -> str | None:
        return self._orientations.get(name)
