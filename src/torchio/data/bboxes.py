"""3D bounding boxes with flexible axis conventions.

Inspired by `torchvision.tv_tensors.BoundingBoxes`, extended to 3D with
support for arbitrary voxel and anatomical axis orderings.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from typing_extensions import Self

from .affine import AffineMatrix
from .axes import AxesType
from .axes import axes_type
from .axes import get_axis_mapping
from .axes import validate_axes


class Representation(Enum):
    """How the six columns of a bounding box are interpreted.

    Attributes:
        CORNERS: Two corners: $(a_1, b_1, c_1, a_2, b_2, c_2)$.
        CENTER_SIZE: Center + size: $(a_c, b_c, c_c, s_a, s_b, s_c)$.
    """

    CORNERS = "corners"
    CENTER_SIZE = "center_size"


class BoundingBoxFormat:
    """Format specification for 3D bounding boxes.

    A format is defined by two components:

    - **axes**: a 3-character string specifying the coordinate system.
      Voxel axes are permutations of `"IJK"`.
      Anatomical axes use one letter from each pair
      `{R, L}`, `{A, P}`, `{S, I}` (e.g., `"RAS"`, `"LPI"`).
    - **representation**: either *corners* (two opposite corners) or
      *center_size* (center point + extent along each axis).

    Args:
        axes: 3-character axis string.
        representation: How the 6 values encode the box.

    Examples:
        >>> from torchio.data.bboxes import BoundingBoxFormat, Representation
        >>> BoundingBoxFormat("IJK", Representation.CORNERS)
        BoundingBoxFormat(axes='IJK', representation='corners')
        >>> BoundingBoxFormat("RAS", "center_size")
        BoundingBoxFormat(axes='RAS', representation='center_size')
    """

    # Predefined convenience formats, set after the class body.
    IJKIJK: BoundingBoxFormat
    IJKWHD: BoundingBoxFormat

    __slots__ = ("_axes", "_representation")

    def __init__(
        self,
        axes: str,
        representation: Representation | str = Representation.CORNERS,
    ) -> None:
        self._axes = validate_axes(axes)
        if isinstance(representation, str):
            representation = Representation(representation)
        self._representation = representation

    @property
    def axes(self) -> str:
        """3-character axis string (e.g., `'IJK'`, `'RAS'`)."""
        return self._axes

    @property
    def representation(self) -> Representation:
        """Corners or center-size."""
        return self._representation

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundingBoxFormat):
            return NotImplemented
        return (
            self._axes == other._axes and self._representation == other._representation
        )

    def __hash__(self) -> int:
        return hash((self._axes, self._representation))

    def __repr__(self) -> str:
        return (
            f"BoundingBoxFormat(axes={self._axes!r},"
            f" representation={self._representation.value!r})"
        )


# --- Predefined formats ---
BoundingBoxFormat.IJKIJK = BoundingBoxFormat("IJK", Representation.CORNERS)
BoundingBoxFormat.IJKWHD = BoundingBoxFormat("IJK", Representation.CENTER_SIZE)


# --- Representation conversion helpers ---


def _corners_to_center_size(data: Tensor) -> Tensor:
    a1, b1, c1, a2, b2, c2 = data.unbind(-1)
    ac = (a1 + a2) / 2
    bc = (b1 + b2) / 2
    cc = (c1 + c2) / 2
    sa = a2 - a1
    sb = b2 - b1
    sc = c2 - c1
    return torch.stack([ac, bc, cc, sa, sb, sc], dim=-1)


def _center_size_to_corners(data: Tensor) -> Tensor:
    ac, bc, cc, sa, sb, sc = data.unbind(-1)
    a1 = ac - sa / 2
    b1 = bc - sb / 2
    c1 = cc - sc / 2
    a2 = ac + sa / 2
    b2 = bc + sb / 2
    c2 = cc + sc / 2
    return torch.stack([a1, b1, c1, a2, b2, c2], dim=-1)


# --- Axis reordering helpers ---


def _permute_corners(
    data: Tensor,
    perm: tuple[int, int, int],
    flips: tuple[bool, bool, bool],
) -> Tensor:
    """Permute and flip columns of a corners-format (N, 6) tensor."""
    p0, p1, p2 = perm
    # Separate the two triplets.
    corner1 = data[:, :3][:, [p0, p1, p2]]
    corner2 = data[:, 3:][:, [p0, p1, p2]]
    # Apply flips (negation).
    for col, flip in enumerate(flips):
        if flip:
            c1 = -corner1[:, col].clone()
            c2 = -corner2[:, col].clone()
            # After negation the min/max may swap, so ensure corner1 < corner2.
            corner1[:, col] = torch.min(c1, c2)
            corner2[:, col] = torch.max(c1, c2)
    return torch.cat([corner1, corner2], dim=-1)


def _permute_center_size(
    data: Tensor,
    perm: tuple[int, int, int],
    flips: tuple[bool, bool, bool],
) -> Tensor:
    """Permute and flip columns of a center-size-format (N, 6) tensor."""
    p0, p1, p2 = perm
    center = data[:, :3][:, [p0, p1, p2]]
    size = data[:, 3:][:, [p0, p1, p2]]
    # Flips negate the center; sizes stay positive.
    for col, flip in enumerate(flips):
        if flip:
            center[:, col] = -center[:, col]
    return torch.cat([center, size], dim=-1)


# --- Voxel ↔ anatomical helpers ---


def _ijk_corners_to_world(
    data: Tensor,
    affine: AffineMatrix,
) -> Tensor:
    """Convert (N, 6) corners from IJK voxel to world (RAS) coordinates."""
    c1 = data[:, :3]
    c2 = data[:, 3:]
    w1 = affine.apply(c1).to(torch.float32)
    w2 = affine.apply(c2).to(torch.float32)
    # After affine, min/max might swap, so normalize.
    lo = torch.min(w1, w2)
    hi = torch.max(w1, w2)
    return torch.cat([lo, hi], dim=-1)


def _world_corners_to_ijk(
    data: Tensor,
    affine: AffineMatrix,
) -> Tensor:
    """Convert (N, 6) corners from world (RAS) to IJK voxel coordinates."""
    inv = affine.inverse()
    c1 = data[:, :3]
    c2 = data[:, 3:]
    v1 = inv.apply(c1).to(torch.float32)
    v2 = inv.apply(c2).to(torch.float32)
    lo = torch.min(v1, v2)
    hi = torch.max(v1, v2)
    return torch.cat([lo, hi], dim=-1)


class BoundingBoxes:
    r"""3D bounding boxes with flexible axis conventions.

    Inspired by `torchvision.tv_tensors.BoundingBoxes`, extended to 3D.
    One instance holds $N$ boxes, each a 6-element vector whose meaning
    is determined by the
    [`format`][torchio.data.bboxes.BoundingBoxFormat].

    Args:
        data: $(N, 6)$ tensor or array.
        format: Interpretation of the 6 columns.
        labels: Optional $(N,)$ integer tensor of class labels per box.
        affine: $4 \times 4$ affine matrix. Identity if not given.
        metadata: Arbitrary metadata dict.

    Examples:
        >>> import torch, torchio as tio
        >>> boxes = tio.BoundingBoxes(
        ...     torch.tensor([[10, 20, 30, 50, 60, 70]]),
        ...     format=tio.BoundingBoxFormat.IJKIJK,
        ... )
        >>> boxes.num_boxes
        1
    """

    def __init__(
        self,
        data: Tensor | npt.ArrayLike,
        *,
        format: BoundingBoxFormat,
        labels: Tensor | None = None,
        affine: AffineMatrix | npt.ArrayLike | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._data = self._parse_data(data)
        self._format = format
        self._labels = self._parse_labels(labels, self._data.shape[0])
        self._affine = self._parse_affine(affine)
        self._metadata: dict[str, Any] = dict(metadata) if metadata else {}

    # --- Parsing ---

    @staticmethod
    def _parse_data(data: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(data, Tensor):
            data = torch.as_tensor(np.asarray(data), dtype=torch.float32)
        if data.ndim != 2 or data.shape[1] != 6:
            msg = f"BoundingBoxes must have shape (N, 6), got {tuple(data.shape)}"
            raise ValueError(msg)
        return data

    @staticmethod
    def _parse_labels(labels: Tensor | None, n: int) -> Tensor | None:
        if labels is None:
            return None
        if labels.shape[0] != n:
            msg = f"Expected {n} labels, got {labels.shape[0]}"
            raise ValueError(msg)
        return labels

    @staticmethod
    def _parse_affine(affine: AffineMatrix | npt.ArrayLike | None) -> AffineMatrix:
        if affine is None:
            return AffineMatrix()
        if isinstance(affine, AffineMatrix):
            return affine
        return AffineMatrix(affine)

    # --- Properties ---

    @property
    def data(self) -> Tensor:
        """$(N, 6)$ tensor of bounding box coordinates."""
        return self._data

    @property
    def format(self) -> BoundingBoxFormat:
        """Interpretation of the 6 columns."""
        return self._format

    @property
    def labels(self) -> Tensor | None:
        """$(N,)$ integer labels, or `None`."""
        return self._labels

    @property
    def affine(self) -> AffineMatrix:
        r"""$4 \times 4$ affine mapping voxel to world coordinates."""
        return self._affine

    @property
    def metadata(self) -> dict[str, Any]:
        """Arbitrary metadata dict."""
        return self._metadata

    @property
    def num_boxes(self) -> int:
        """Number of bounding boxes."""
        return self._data.shape[0]

    @property
    def device(self) -> torch.device:
        """Device the bounding box data resides on."""
        return self._data.device

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move bounding box data to a device and/or cast to a dtype.

        Returns:
            `self` (modified in-place).
        """
        self._data = self._data.to(*args, **kwargs)
        if self._labels is not None:
            self._labels = self._labels.to(*args, **kwargs)
        return self

    # --- Methods ---

    def to_format(self, format: BoundingBoxFormat) -> Self:
        """Convert to a different bounding box format.

        Handles representation changes (corners ↔ center-size), axis
        permutations within the same type, and voxel ↔ anatomical
        conversions (using the stored affine). The affine maps IJK
        voxel indices to RAS+ world coordinates, regardless of the
        image's voxel orientation.

        Args:
            format: Target format.

        Returns:
            New `BoundingBoxes` in the target format.
        """
        if format == self._format:
            return self._clone(format=format)

        src_axes = self._format.axes
        tgt_axes = format.axes
        src_repr = self._format.representation
        tgt_repr = format.representation

        src_type = axes_type(src_axes)
        tgt_type = axes_type(tgt_axes)

        # Step 1: normalise to corners in source axes.
        data = self._data
        if src_repr == Representation.CENTER_SIZE:
            data = _center_size_to_corners(data)

        # Step 2: axis conversion (now in corners).
        if src_axes != tgt_axes:
            if src_type == tgt_type:
                # Same family: permute + flip.
                perm, flips = get_axis_mapping(src_axes, tgt_axes)
                data = _permute_corners(data, perm, flips)
            else:
                # Cross-type: go through world coordinates.
                data = self._cross_type_corners(
                    data,
                    src_axes,
                    src_type,
                    tgt_axes,
                    tgt_type,
                )

        # Step 3: convert to target representation.
        if tgt_repr == Representation.CENTER_SIZE:
            data = _corners_to_center_size(data)

        return self._clone(data=data, format=format)

    def new_like(
        self,
        *,
        data: Tensor | npt.ArrayLike,
        labels: Tensor | None = None,
        affine: AffineMatrix | npt.ArrayLike | None = None,
    ) -> Self:
        """Create new BoundingBoxes with the same format and metadata.

        Args:
            data: New $(N, 6)$ coordinates.
            labels: New labels. If `None`, no labels.
            affine: New affine. If `None`, uses `self.affine`.
        """
        new_affine = (
            self._parse_affine(affine) if affine is not None else self._affine.clone()
        )
        return type(self)(
            data,
            format=self._format,
            labels=labels,
            affine=new_affine,
            metadata=dict(self._metadata),
        )

    # --- Internal ---

    def _clone(
        self,
        *,
        data: Tensor | None = None,
        format: BoundingBoxFormat | None = None,
    ) -> Self:
        return type(self)(
            data if data is not None else self._data.clone(),
            format=format if format is not None else self._format,
            labels=self._labels.clone() if self._labels is not None else None,
            affine=self._affine.clone(),
            metadata=dict(self._metadata),
        )

    def _cross_type_corners(
        self,
        data: Tensor,
        src_axes: str,
        src_type: AxesType,
        tgt_axes: str,
        tgt_type: AxesType,
    ) -> Tensor:
        """Convert corners between voxel and anatomical coordinate systems."""
        if src_type == AxesType.VOXEL:
            # Voxel → RAS via affine, then optionally reorder/flip within
            # anatomical.
            # First normalise voxel order to IJK.
            if src_axes != "IJK":
                perm, _ = get_axis_mapping(src_axes, "IJK")
                data = _permute_corners(data, perm, (False, False, False))
            # Apply affine to get RAS.
            data = _ijk_corners_to_world(data, self._affine)
            if tgt_axes != "RAS":
                perm, flips = get_axis_mapping("RAS", tgt_axes)
                data = _permute_corners(data, perm, flips)
        else:
            # Anatomical → voxel.
            # Normalise to RAS before applying the inverse affine.
            if src_axes != "RAS":
                perm, flips = get_axis_mapping(src_axes, "RAS")
                data = _permute_corners(data, perm, flips)
            # World → IJK via inverse affine.
            data = _world_corners_to_ijk(data, self._affine)
            # Reorder to target voxel axes if needed.
            if tgt_axes != "IJK":
                perm, _ = get_axis_mapping("IJK", tgt_axes)
                data = _permute_corners(data, perm, (False, False, False))
        return data

    # --- Dunder ---

    def __len__(self) -> int:
        return self.num_boxes

    def __repr__(self) -> str:
        return (
            f"BoundingBoxes(num_boxes={self.num_boxes},"
            f" axes={self._format.axes!r},"
            f" representation={self._format.representation.value!r})"
        )

    def __deepcopy__(self, memo: dict) -> Self:
        new = type(self)(
            self._data.clone(),
            format=self._format,
            labels=self._labels.clone() if self._labels is not None else None,
            affine=self._affine.clone(),
            metadata=dict(self._metadata),
        )
        memo[id(self)] = new
        return new
