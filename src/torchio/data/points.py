"""Points class for storing sets of 3D coordinates."""

from __future__ import annotations

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


class Points:
    """A set of 3D points with a named axis convention.

    Stores an $(N, 3)$ tensor of coordinates alongside an affine matrix
    and an axis string describing the coordinate system.

    The default axis convention is `"IJK"` (voxel indices). Points can
    be converted to any other axis convention (including anatomical
    systems like `"RAS"` or `"LPI"`) via
    [`to_axes`][torchio.data.points.Points.to_axes].

    Args:
        data: $(N, 3)$ tensor or array of point coordinates.
        axes: 3-character axis string (default `"IJK"`).
        affine: $4 \\times 4$ affine matrix. Identity if not given.
        metadata: Arbitrary metadata dict.

    Examples:
        >>> import torch, torchio as tio
        >>> pts = tio.Points(torch.tensor([[10.0, 20.0, 30.0]]))
        >>> pts.axes
        'IJK'
        >>> pts.num_points
        1
    """

    def __init__(
        self,
        data: Tensor | npt.ArrayLike,
        *,
        axes: str = "IJK",
        affine: AffineMatrix | npt.ArrayLike | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._data = self._parse_data(data)
        self._axes = validate_axes(axes)
        self._affine = self._parse_affine(affine)
        self._metadata: dict[str, Any] = dict(metadata) if metadata else {}

    # --- Parsing ---

    @staticmethod
    def _parse_data(data: Tensor | npt.ArrayLike) -> Tensor:
        if not isinstance(data, Tensor):
            data = torch.as_tensor(np.asarray(data), dtype=torch.float32)
        if data.ndim != 2 or data.shape[1] != 3:
            msg = f"Points must have shape (N, 3), got {tuple(data.shape)}"
            raise ValueError(msg)
        return data

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
        """$(N, 3)$ tensor of point coordinates."""
        return self._data

    @property
    def axes(self) -> str:
        """3-character axis string (e.g., `'IJK'`, `'RAS'`)."""
        return self._axes

    @property
    def affine(self) -> AffineMatrix:
        """$4 \\times 4$ affine mapping voxel to world coordinates."""
        return self._affine

    @property
    def metadata(self) -> dict[str, Any]:
        """Arbitrary metadata dict."""
        return self._metadata

    @property
    def num_points(self) -> int:
        """Number of points."""
        return self._data.shape[0]

    @property
    def device(self) -> torch.device:
        """Device the point data resides on."""
        return self._data.device

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move point data to a device and/or cast to a dtype.

        Returns:
            `self` (modified in-place).
        """
        self._data = self._data.to(*args, **kwargs)
        return self

    # --- Methods ---

    def to_world(self) -> Tensor:
        """Return points in RAS+ world (mm) coordinates.

        Equivalent to `self.to_axes("RAS").data`, cast to `float32`.
        The stored axis convention is respected, so points already in
        anatomical coordinates are not transformed by the affine again.

        Returns:
            $(N, 3)$ float32 tensor in RAS+ world coordinates.
        """
        return self.to_axes("RAS").data.to(torch.float32)

    def to_axes(self, target: str) -> Self:
        """Convert points to a different axis convention.

        Handles permutations within the same type (voxel ↔ voxel,
        anatomical ↔ anatomical) and cross-type conversions
        (voxel ↔ anatomical) using the stored affine. The affine maps
        IJK voxel indices to RAS+ world coordinates, regardless of the
        image's voxel orientation.

        Args:
            target: Target axis string.

        Returns:
            New `Points` in the target axis convention.
        """
        target = validate_axes(target)
        if target == self._axes:
            return self._clone(axes=target)

        src_type = axes_type(self._axes)
        tgt_type = axes_type(target)

        if src_type == tgt_type:
            perm, flips = get_axis_mapping(self._axes, target)
            converted = self._permute_and_flip(self._data, perm, flips)
        else:
            converted = self._cross_type(self._data, src_type, target, tgt_type)

        return self._clone(data=converted, axes=target)

    def new_like(
        self,
        *,
        data: Tensor | npt.ArrayLike,
        affine: AffineMatrix | npt.ArrayLike | None = None,
    ) -> Self:
        """Create a new Points with the same metadata and axes.

        Args:
            data: New $(N, 3)$ coordinates.
            affine: New affine. If `None`, uses `self.affine`.
        """
        new_affine = (
            self._parse_affine(affine) if affine is not None else self._affine.clone()
        )
        return type(self)(
            data,
            axes=self._axes,
            affine=new_affine,
            metadata=dict(self._metadata),
        )

    # --- Internal ---

    def _clone(
        self,
        *,
        data: Tensor | None = None,
        axes: str | None = None,
    ) -> Self:
        return type(self)(
            data if data is not None else self._data.clone(),
            axes=axes if axes is not None else self._axes,
            affine=self._affine.clone(),
            metadata=dict(self._metadata),
        )

    @staticmethod
    def _permute_and_flip(
        data: Tensor,
        perm: tuple[int, int, int],
        flips: tuple[bool, bool, bool],
    ) -> Tensor:
        result = data[:, list(perm)]
        for col, flip in enumerate(flips):
            if flip:
                result[:, col] = -result[:, col]
        return result

    def _cross_type(
        self,
        data: Tensor,
        src_type: AxesType,
        tgt_axes: str,
        tgt_type: AxesType,
    ) -> Tensor:
        if src_type == AxesType.VOXEL:
            # Voxel → anatomical.
            # Normalise to IJK first.
            if self._axes != "IJK":
                perm, _ = get_axis_mapping(self._axes, "IJK")
                data = data[:, list(perm)]
            # Apply affine → world.
            world = self._affine.apply(data).to(torch.float32)
            if tgt_axes != "RAS":
                perm, flips = get_axis_mapping("RAS", tgt_axes)
                world = self._permute_and_flip(world, perm, flips)
            return world
        else:
            # Anatomical → voxel.
            # Normalise to RAS before applying the inverse affine.
            if self._axes != "RAS":
                perm, flips = get_axis_mapping(self._axes, "RAS")
                data = self._permute_and_flip(data, perm, flips)
            # Inverse affine → IJK.
            inv = self._affine.inverse()
            ijk = inv.apply(data).to(torch.float32)
            # Reorder to target voxel axes.
            if tgt_axes != "IJK":
                perm, _ = get_axis_mapping("IJK", tgt_axes)
                ijk = ijk[:, list(perm)]
            return ijk

    # --- Dunder ---

    def __len__(self) -> int:
        return self.num_points

    def __repr__(self) -> str:
        return f"Points(num_points={self.num_points}, axes={self._axes!r})"

    def __deepcopy__(self, memo: dict) -> Self:
        new = type(self)(
            self._data.clone(),
            axes=self._axes,
            affine=self._affine.clone(),
            metadata=dict(self._metadata),
        )
        memo[id(self)] = new
        return new
