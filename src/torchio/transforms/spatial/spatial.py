"""Unified spatial transforms.

Combines resampling, affine motion, and elastic deformation into a single
`grid_sample` call.  The public API consists of four classes:

- [`Spatial`][torchio.Spatial]: the unified transform.
- [`Resample`][torchio.Resample]: resampling-only convenience wrapper.
- [`Affine`][torchio.Affine]: affine-only convenience wrapper.
- [`ElasticDeformation`][torchio.ElasticDeformation]: elastic-only convenience wrapper.

The module-level helpers handle coordinate math, grid construction,
serialization for history replay, and parameter parsing/validation.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any
from typing import Literal
from typing import TypeAlias
from typing import TypeGuard
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
from einops import rearrange
from torch import Tensor
from torch.distributions import Distribution

from ...data.affine import AffineMatrix
from ...data.batch import ImagesBatch
from ...data.batch import SubjectsBatch
from ...data.image import Image
from ...data.image import LabelMap
from ...data.image import ScalarImage
from ...types import TypeSpacing
from ...types import TypeThreeInts
from ..parameter_range import Choice
from ..parameter_range import _ParameterRange
from ..transform import SpatialTransform


@dataclass
class _PerSampleGrids:
    """Per-element spatial parameters for per-instance augmentation.

    Each list has one entry per batch element. A `None` affine matrix
    and `None` control-point field produce an identity grid for that
    element (used for elements gated out by per-element probability).

    Attributes:
        affine_matrices: Per-element world-space affine matrices.
        control_points: Per-element elastic control-point fields.
        max_displacements: Per-element maximum displacements.
    """

    affine_matrices: list[np.ndarray | None]
    control_points: list[Tensor | None]
    max_displacements: list[tuple[float, float, float] | None]


TypeParameterValue: TypeAlias = (
    int
    | float
    | tuple[int | float]
    | tuple[int | float, int | float]
    | tuple[int | float, int | float, int | float]
    | tuple[
        int | float,
        int | float,
        int | float,
        int | float,
        int | float,
        int | float,
    ]
    | tuple  # per-axis mixed specs like (0, Choice([...]), (-10, 10))
    | Choice
    | Distribution
)
TypeTarget: TypeAlias = (
    int
    | float
    | TypeSpacing
    | tuple[float, float]  # uniform spacing range, per axis
    | tuple[float, float, float, float, float, float]  # per-axis spacing ranges
    | Choice
    | Distribution
    | str
    | Path
    | Image
    | tuple[Sequence[int], AffineMatrix | Tensor | npt.ArrayLike]
    | None
)
#: The spacing forms of `target` that resolve to a (possibly random) spacing.
TypeSpacingSpec: TypeAlias = (
    int | float | tuple | list | np.ndarray | Choice | Distribution
)
TypeControlPoints: TypeAlias = Tensor | npt.ArrayLike
TypeTargetSpace: TypeAlias = tuple[TypeThreeInts, AffineMatrix]
TypeImageInterpolation: TypeAlias = Literal[
    "nearest",
    "linear",
    "quadratic",
    "cubic",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
]
#: Label maps additionally accept the partial-volume-aware `"label"` mode.
TypeLabelInterpolation: TypeAlias = TypeImageInterpolation | Literal["label"]
#: Broad alias accepting any interpolation mode, including `"label"`. Used by
#: internal helpers that handle both image and label interpolation.
TypeInterpolation: TypeAlias = TypeLabelInterpolation
TypeCenter: TypeAlias = Literal["image", "origin"]
TypePadValue: TypeAlias = Literal["minimum", "mean", "otsu"]

#: Partial-volume label interpolation mode. Only valid for label maps: the map
#: is one-hot encoded, each channel is resampled linearly, and the per-voxel
#: argmax recovers the discrete labels (see `_resample_label_partial_volume`).
LABEL_INTERPOLATION = "label"

_SUPPORTED_INTERPOLATIONS = (
    "nearest",
    "linear",
    "quadratic",
    "cubic",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    LABEL_INTERPOLATION,
)
_INTERPOLATION_TO_ORDER: dict[str, int] = {
    "nearest": 0,
    "linear": 1,
    "quadratic": 2,
    "cubic": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
}
_TORCH_INTERPOLATION_MODE = {
    "nearest": "nearest",
    "linear": "bilinear",
}
_SUPPORTED_PAD_VALUES = ("minimum", "mean", "otsu")
_SPLINE_ORDER = 3


class Spatial(SpatialTransform):
    r"""Apply resampling, affine motion, and elastic deformation together.

    This transform can:

    1. resample to a new space,
    2. apply a global affine mapping, and
    3. apply a dense elastic field,

    using a single sampling grid.

    The convenience wrappers [`Resample`][torchio.Resample],
    [`Affine`][torchio.Affine], and
    [`ElasticDeformation`][torchio.ElasticDeformation] expose subsets
    of these parameters with sensible defaults.

    Args:
        target: Output space.  Can be one of:

            - A scalar or 3-tuple of floats: output voxel spacing in mm.
              E.g., `1` for 1 mm isotropic, `(0.5, 0.5, 2.0)` for
              anisotropic.
            - A random spacing spec, sampled once per call at apply time: a
              2-tuple `(lo, hi)` (uniform, per axis), a 6-tuple
              `(lo1, hi1, lo2, hi2, lo3, hi3)` (per-axis ranges), a `Choice`,
              or a `torch.distributions.Distribution`. E.g.,
              `target=(1, 2)` or `target=(2, 4, 2, 4, 3, 6)`.
            - A `str`: either a path to an image file, or the name of
              an image in the subject (e.g., `"t1"`).
            - An [`Image`][torchio.Image] instance.
            - A `(spatial_shape, affine)` pair.
            - `None` (default): the output grid matches the input grid.
        scales: Scale factors $(s_1, s_2, s_3)$ for each axis.
            If a single value $x$ is given, all axes use $x$.
            If two values $(a, b)$ are given,
            $s_i \sim \mathcal{U}(a, b)$.
            If six values $(a_1, b_1, a_2, b_2, a_3, b_3)$ are given,
            $s_i \sim \mathcal{U}(a_i, b_i)$ independently.
            A `torch.distributions.Distribution` may also be passed.
            For example, `scales=0.5` halves the apparent object
            size (zoom out), and `scales=2` doubles it (zoom in).
        degrees: Euler rotation angles $(\theta_1, \theta_2, \theta_3)$
            in degrees, following the same value/range/distribution
            convention as *scales*.
        translation: Translation $(t_1, t_2, t_3)$ in mm, following
            the same convention.  The direction depends on the image
            orientation: in RAS+, `translation=(10, 0, 0)` shifts
            10 mm to the right.
        isotropic: If `True`, sample a single scale factor and
            reuse it for all three axes.  *scales* must then be a
            scalar or 2-value range.
        center: Pivot point for rotation and scaling.
            `"image"` (default) uses the image center;
            `"origin"` uses the world-coordinate origin.
        control_points: Optional pre-computed coarse displacement
            field with shape `(n_i, n_j, n_k, 3)` in mm.  If given,
            *num_control_points*, *max_displacement*, and
            *locked_borders* are ignored.
        num_control_points: Number of control points along each
            dimension of the coarse grid.  Can be a single `int`
            (isotropic) or a 3-tuple.  Minimum is 4.  Smaller values
            produce smoother deformations.
        max_displacement: Maximum displacement at each control point,
            in mm.  Follows the same value/range/distribution
            convention as *scales*.  Zero (default) disables elastic
            deformation.
        locked_borders: Number of outer control-point layers whose
            displacement is forced to zero.  `0` keeps all
            displacements; `1` zeros the outermost layer; `2`
            (default) zeros the two outermost layers.
        affine_first: If `True` (default), apply the affine mapping
            before the elastic field.  If `False`, apply the elastic
            field first.  The difference is significant for large
            transforms.
        image_interpolation: Interpolation for
            [`ScalarImage`][torchio.ScalarImage] instances.  `"linear"`
            (default) or `"nearest"` use a fast path; higher-order
            B-spline modes `"quadratic"`, `"cubic"`, `"fourth"`,
            `"fifth"`, `"sixth"`, and `"seventh"` are also supported, as
            are the equivalent integer orders `0`-`7`.
        label_interpolation: Interpolation for
            [`LabelMap`][torchio.LabelMap] instances.  Accepts the same
            values as *image_interpolation* (`"nearest"` is the default),
            plus the special `"label"` mode.  The `"label"` mode performs
            partial-volume-aware resampling: the label map is one-hot
            encoded, each channel is resampled with linear interpolation,
            and the per-voxel argmax recovers the discrete labels.
            Compared with `"nearest"`, this reduces staircase artifacts
            and yields more accurate label volumes, which is especially
            useful when downsampling.  For single-channel label maps it
            never invents intermediate label values that were absent from
            the input (the only new value that can appear is
            `default_pad_label`, used for out-of-bounds voxels).  A
            multi-channel (already one-hot or probabilistic) label map is
            instead resampled per channel, so its output can contain
            fractional partial volumes.

            The `"label"` mode is more memory- and compute-intensive
            because it processes one channel per label, so memory scales
            with the number of distinct labels.  For maps with many labels
            (e.g. hundreds of brain structures) prefer processing patch by
            patch with [`GridSampler`][torchio.GridSampler] and
            [`PatchAggregator`][torchio.PatchAggregator] to bound memory.
        one_hot_label_interpolation: Interpolation used for the one-hot
            channels of the `"label"` mode.  Accepts the same values as
            *image_interpolation* (`"nearest"`, `"linear"`, or the
            higher-order B-spline modes / integer orders 0-7), but **not**
            `"label"`.  Defaults to `"linear"`.  Higher-order modes give
            smoother label boundaries, which is particularly useful when
            upsampling.  Ignored unless `label_interpolation="label"`.
        antialias: If `True`, apply Gaussian smoothing before
            downsampling intensity images.  Label maps are smoothed only
            when `label_interpolation="label"` (the one-hot channels are
            blurred before downsampling); otherwise they are left
            untouched.  The standard deviations follow
            [Cardoso et al., MICCAI 2015](https://link.springer.com/chapter/10.1007/978-3-319-24571-3_81).
        default_pad_value: Fill rule for out-of-bounds intensity
            voxels.  `"minimum"` (default), `"mean"`, `"otsu"`,
            or a numeric value.
        default_pad_label: Numeric fill value for out-of-bounds label
            voxels.
        **kwargs: See [`Transform`][torchio.Transform].

    Note:
        The randomizable parameters (`scales`, `degrees`, `translation`,
        `max_displacement`, and the spacing form of `target`) follow a common
        value/range/distribution convention: a scalar is deterministic, a
        2-tuple $(a, b)$ samples uniformly, a 3-tuple sets per-axis values, a
        6-tuple sets per-axis ranges, and a `Choice` or
        `torch.distributions.Distribution` samples from a discrete set or a
        distribution. Structural parameters (`center`, interpolation, padding)
        are not randomizable.

    Examples:
        >>> import torchio as tio
        >>> # Resample to 1 mm isotropic with a random rotation
        >>> transform = tio.Spatial(
        ...     target=1,
        ...     degrees=(-10, 10),
        ...     translation=(-5, 5),
        ... )
        >>> # Elastic deformation only
        >>> transform = tio.Spatial(
        ...     max_displacement=7.5,
        ...     num_control_points=7,
        ... )
        >>> transformed = transform(subject)
    """

    def __init__(
        self,
        *,
        target: TypeTarget = None,
        scales: TypeParameterValue = 1.0,
        degrees: TypeParameterValue = 0.0,
        translation: TypeParameterValue = 0.0,
        isotropic: bool = False,
        center: TypeCenter = "image",
        control_points: TypeControlPoints | None = None,
        num_control_points: int | TypeThreeInts = 7,
        max_displacement: TypeParameterValue = 0.0,
        locked_borders: int = 2,
        affine_first: bool = True,
        image_interpolation: TypeImageInterpolation | int = "linear",
        label_interpolation: TypeLabelInterpolation | int = "nearest",
        one_hot_label_interpolation: TypeImageInterpolation | int = "linear",
        antialias: bool = False,
        default_pad_value: TypePadValue | float = "minimum",
        default_pad_label: int | float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.target = target
        _validate_isotropic(scales, isotropic)
        self.scales = _to_positive_range(scales)
        self.degrees = _to_parameter_range(degrees)
        self.translation = _to_parameter_range(translation)
        self.isotropic = isotropic
        self.center = _parse_center(center)
        self.control_points = (
            _parse_control_points(control_points)
            if control_points is not None
            else None
        )
        self.num_control_points = _parse_num_control_points(num_control_points)
        self.max_displacement = _to_nonnegative_parameter_range(max_displacement)
        self.locked_borders = _parse_locked_borders(locked_borders)
        if self.locked_borders == 2 and 4 in self.num_control_points:
            msg = (
                "locked_borders=2 with 4 control points along any axis yields an"
                " identity elastic field"
            )
            raise ValueError(msg)
        self.affine_first = affine_first
        parsed_image_interpolation = _parse_interpolation(image_interpolation)
        if parsed_image_interpolation == LABEL_INTERPOLATION:
            msg = (
                f'image_interpolation cannot be "{LABEL_INTERPOLATION}"; that mode'
                " is only valid for label_interpolation"
            )
            raise ValueError(msg)
        self.image_interpolation = parsed_image_interpolation
        self.label_interpolation = _parse_interpolation(label_interpolation)
        self.one_hot_label_interpolation = _parse_one_hot_label_interpolation(
            one_hot_label_interpolation,
        )
        self.antialias = antialias
        self.default_pad_value = _parse_default_pad_value(default_pad_value)
        if not isinstance(default_pad_label, Number):
            msg = f"default_pad_label must be numeric, got {type(default_pad_label)}"
            raise TypeError(msg)
        self.default_pad_label = float(default_pad_label)

    @property
    def supports_per_instance_params(self) -> bool:
        return True

    @property
    def supports_per_instance_p(self) -> bool:
        # Per-element gating requires a shape-preserving transform so the
        # gated-out element is a true no-op. A resampling target changes
        # the grid, so keep batch-wide gating in that case.
        return self.target is None

    def _sample_one(
        self,
        shape: TypeThreeInts,
        affine: AffineMatrix,
    ) -> tuple[
        np.ndarray | None,
        Tensor | None,
        tuple[float, float, float] | None,
        bool,
    ]:
        """Sample one set of affine and elastic parameters.

        Args:
            shape: Spatial shape used to build the affine and to suppress
                out-of-plane components for 2D inputs.
            affine: Affine of the reference image.

        Returns:
            A tuple `(forward_affine, control_points, max_displacement,
            has_geometry)` where `has_geometry` is `True` when either an
            affine or an elastic component is present.
        """
        sampled_scales = _sample_scales(self.scales, self.isotropic)
        sampled_degrees = self.degrees.sample()
        sampled_translation = self.translation.sample()
        has_affine = _has_affine_component(
            sampled_scales,
            sampled_degrees,
            sampled_translation,
        )
        control_points, max_displacement = _resolve_control_points(
            self.control_points,
            self.num_control_points,
            self.max_displacement,
            self.locked_borders,
        )
        has_elastic = control_points is not None
        forward_affine = None
        if has_affine:
            forward_affine = _build_forward_affine(
                scales=sampled_scales,
                degrees=sampled_degrees,
                translation=sampled_translation,
                center=self.center,
                shape=shape,
                affine=affine,
            )
        return (
            forward_affine,
            control_points,
            max_displacement,
            (has_affine or has_elastic),
        )

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample random parameters and resolve the output space.

        Scales, degrees, translation, and control-point displacements are
        sampled per batch element when per-instance augmentation is
        active (the default for batches), and once otherwise.

        Returns:
            Dict of serializable parameters for `apply_transform` and
            history replay.
        """
        images = self._get_images(batch)
        if not images:
            return {"selected_images": []}

        _, first_batch = next(iter(images.items()))
        first_shape = _get_spatial_shape(first_batch)
        first_affine = first_batch.affines[0]

        params: dict[str, Any] = {
            "selected_images": list(images.keys()),
            "original": _serialize_space((first_shape, first_affine)),
            "affine_first": self.affine_first,
            "image_interpolation": self.image_interpolation,
            "label_interpolation": self.label_interpolation,
            "one_hot_label_interpolation": self.one_hot_label_interpolation,
            "antialias": self.antialias,
            "default_pad_value": self.default_pad_value,
            "default_pad_label": self.default_pad_label,
        }

        n = self._resolve_n(batch)
        if n is None:
            forward_affine, control_points, max_displacement, has_geometry = (
                self._sample_one(first_shape, first_affine)
            )
            if has_geometry:
                _check_shared_space(images, first_shape, first_affine)
            # Resolve the (possibly random) target after sampling the
            # geometry so the RNG stream matches the batch-shared path.
            target_space = _resolve_target_space(
                self.target,
                batch,
                first_shape,
                first_affine,
            )
            params["target"] = _serialize_space(target_space)
            params["affine_matrix"] = _serialize_matrix(forward_affine)
            params["control_points"] = _serialize_control_points(control_points)
            params["max_displacement"] = (
                list(max_displacement) if max_displacement else None
            )
            return params

        keep = self._keep_mask(batch, n)
        affine_list, control_points_list, displacement_list, any_geometry = (
            self._sample_per_element_geometry(n, first_shape, first_affine, keep)
        )
        if any_geometry:
            _check_shared_space(images, first_shape, first_affine)
        target_space = _resolve_target_space(
            self.target,
            batch,
            first_shape,
            first_affine,
        )
        params["target"] = _serialize_space(target_space)
        params["affine_matrix"] = affine_list
        params["control_points"] = control_points_list
        params["max_displacement"] = displacement_list
        self._tag_batched(
            params,
            batch,
            n,
            keep,
            ["affine_matrix", "control_points", "max_displacement"],
        )
        return params

    def _sample_per_element_geometry(
        self,
        n: int,
        first_shape: TypeThreeInts,
        first_affine: AffineMatrix,
        keep: Tensor | None,
    ) -> tuple[list[Any], list[Any], list[Any], bool]:
        """Sample serializable geometry for each batch element.

        Gated-out elements (``keep[index]`` is false) contribute ``None``
        placeholders so the per-element lists stay aligned with the batch.

        Args:
            n: Number of batch elements.
            first_shape: Spatial shape of the first image.
            first_affine: Affine of the first image.
            keep: Per-element keep mask, or ``None`` to keep all elements.

        Returns:
            The affine, control-point and max-displacement lists plus a
            flag indicating whether any element produced a geometry
            transform.
        """
        affine_list: list[Any] = []
        control_points_list: list[Any] = []
        displacement_list: list[Any] = []
        any_geometry = False
        for index in range(n):
            kept = keep is None or bool(keep[index])
            if not kept:
                affine_list.append(None)
                control_points_list.append(None)
                displacement_list.append(None)
                continue
            forward_affine, control_points, max_displacement, has_geometry = (
                self._sample_one(first_shape, first_affine)
            )
            any_geometry = any_geometry or has_geometry
            affine_list.append(_serialize_matrix(forward_affine))
            control_points_list.append(_serialize_control_points(control_points))
            displacement_list.append(
                list(max_displacement) if max_displacement else None
            )
        return affine_list, control_points_list, displacement_list, any_geometry

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Apply the spatial mapping to every selected image in *batch*.

        One sampling grid is built per batch element when per-instance
        parameters are present, and a single shared grid otherwise.
        """
        selected_images = params.get("selected_images", [])
        if not selected_images:
            return batch

        target_space = _deserialize_space(params["target"])

        affine_matrix, control_points, max_displacement, per_sample = (
            _resolve_spatial_params(params)
        )
        is_noop = (
            target_space is None
            and affine_matrix is None
            and control_points is None
            and max_displacement is None
            and per_sample is None
        )
        if is_noop:
            # A true no-op (no resampling and no geometry, e.g. every
            # element gated out) must leave the data and the per-sample
            # affines untouched instead of rebuilding an identity grid.
            return batch
        _apply_spatial_to_batch(
            batch=batch,
            image_names=selected_images,
            target_space=target_space,
            affine_matrix=affine_matrix,
            control_points=control_points,
            max_displacement=max_displacement,
            affine_first=params["affine_first"],
            image_interpolation=params["image_interpolation"],
            label_interpolation=params["label_interpolation"],
            one_hot_label_interpolation=params.get(
                "one_hot_label_interpolation",
                "linear",
            ),
            antialias=params.get("antialias", False),
            default_pad_value=params["default_pad_value"],
            default_pad_label=float(params["default_pad_label"]),
            per_sample=per_sample,
        )
        return batch

    @property
    def invertible(self) -> bool:
        """Whether this transform can be inverted."""
        return True

    def inverse(self, params: dict[str, Any]) -> _SpatialInverse:
        """Build the inverse transform from recorded parameters.

        The affine component is inverted exactly.  The elastic component
        is approximated by negating the sampled displacement field.  The
        `affine_first` flag is flipped so that the inverse operations
        run in the opposite order. Per-instance parameters are inverted
        element by element.

        Args:
            params: The parameter dict produced by `make_params`.

        Returns:
            A `_SpatialInverse` that resamples back to the original grid.
        """
        original_space = _deserialize_space(params["original"])
        if original_space is None:
            msg = "Spatial inverse needs the original output space"
            raise RuntimeError(msg)

        common: dict[str, Any] = {
            "target": original_space,
            "affine_first": not params["affine_first"],
            "image_interpolation": params["image_interpolation"],
            "label_interpolation": params["label_interpolation"],
            "one_hot_label_interpolation": params.get(
                "one_hot_label_interpolation",
                "linear",
            ),
            "default_pad_value": params["default_pad_value"],
            "default_pad_label": float(params["default_pad_label"]),
            "copy": False,
            # Invert only the images the forward pass actually transformed,
            # so excluded images (e.g. label maps) are not resampled.
            "include": params["selected_images"],
        }

        batched_keys = params.get("_batched_keys") or []
        if "affine_matrix" in batched_keys:
            per_sample = _invert_per_sample(params)
            return _SpatialInverse(
                affine_matrix=None,
                control_points=None,
                per_sample=per_sample,
                **common,
            )

        affine_matrix = _deserialize_matrix(params["affine_matrix"])
        inverse_affine = None
        if affine_matrix is not None:
            inverse_affine = np.linalg.inv(affine_matrix)
        control_points = _deserialize_control_points(params["control_points"])
        inverse_control_points = None
        if control_points is not None:
            inverse_control_points = -control_points
        return _SpatialInverse(
            affine_matrix=inverse_affine,
            control_points=inverse_control_points,
            **common,
        )


class _SpatialInverse(SpatialTransform):
    """Concrete inverse of [`Spatial`][torchio.Spatial], used for history replay.

    Stores the exact inverse affine matrix, the negated elastic field,
    and the original output space so that `apply_inverse_transform`
    can restore the geometry of images transformed by [`Spatial`][torchio.Spatial].
    """

    def __init__(
        self,
        *,
        target: TypeTargetSpace,
        affine_matrix: npt.ArrayLike | None,
        control_points: TypeControlPoints | None,
        affine_first: bool,
        image_interpolation: TypeImageInterpolation,
        label_interpolation: TypeLabelInterpolation,
        one_hot_label_interpolation: TypeImageInterpolation = "linear",
        default_pad_value: TypePadValue | float,
        default_pad_label: float,
        per_sample: _PerSampleGrids | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.target = target
        self.affine_matrix = (
            np.asarray(affine_matrix, dtype=np.float64).copy()
            if affine_matrix is not None
            else None
        )
        self.control_points = (
            _parse_control_points(control_points)
            if control_points is not None
            else None
        )
        self.per_sample = per_sample
        self.affine_first = affine_first
        self.image_interpolation = cast(
            TypeImageInterpolation,
            _parse_interpolation(image_interpolation),
        )
        self.label_interpolation = _parse_interpolation(label_interpolation)
        self.one_hot_label_interpolation = _parse_one_hot_label_interpolation(
            one_hot_label_interpolation,
        )
        self.default_pad_value = _parse_default_pad_value(default_pad_value)
        self.default_pad_label = float(default_pad_label)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Return empty params; all state is stored in instance attributes."""
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Resample back to the recorded original space."""
        max_displacement = None
        if self.per_sample is None and self.control_points is not None:
            max_displacement = _max_abs_displacement(self.control_points)
        _apply_spatial_to_batch(
            batch=batch,
            image_names=list(self._get_images(batch)),
            target_space=self.target,
            affine_matrix=self.affine_matrix,
            control_points=self.control_points,
            max_displacement=max_displacement,
            affine_first=self.affine_first,
            image_interpolation=self.image_interpolation,
            label_interpolation=self.label_interpolation,
            one_hot_label_interpolation=self.one_hot_label_interpolation,
            antialias=False,
            default_pad_value=self.default_pad_value,
            default_pad_label=self.default_pad_label,
            per_sample=self.per_sample,
        )
        return batch


class Resample(Spatial):
    r"""Resample images to a different space.

    Convenience wrapper around [`Spatial`][torchio.Spatial] exposing
    only the resampling parameters.

    Args:
        target: Output space (see [`Spatial`][torchio.Spatial]).
            Defaults to 1 mm isotropic.
        image_interpolation: See [`Spatial`][torchio.Spatial].
        label_interpolation: See [`Spatial`][torchio.Spatial].
        one_hot_label_interpolation: See [`Spatial`][torchio.Spatial].
        antialias: See [`Spatial`][torchio.Spatial].
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Resample(2)               # 2 mm isotropic
        >>> transform = tio.Resample("t1")            # match "t1" space
        >>> transform = tio.Resample((1, 1, 3))       # anisotropic
        >>> # Partial-volume-aware label resampling
        >>> transform = tio.Resample(
        ...     2,
        ...     label_interpolation="label",
        ...     antialias=True,
        ... )
    """

    def __init__(
        self,
        target: TypeTarget = 1,
        image_interpolation: TypeImageInterpolation | int = "linear",
        label_interpolation: TypeLabelInterpolation | int = "nearest",
        one_hot_label_interpolation: TypeImageInterpolation | int = "linear",
        antialias: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            target=target,
            image_interpolation=image_interpolation,
            label_interpolation=label_interpolation,
            one_hot_label_interpolation=one_hot_label_interpolation,
            antialias=antialias,
            **kwargs,
        )


class Affine(Spatial):
    r"""Apply a random or fixed affine transform.

    Convenience wrapper around [`Spatial`][torchio.Spatial] exposing
    only the affine parameters.  The affine matrix data structure is
    available as [`AffineMatrix`][torchio.AffineMatrix].

    Args:
        scales: See [`Spatial`][torchio.Spatial].
            Default: `1.0` (no scaling).
        degrees: See [`Spatial`][torchio.Spatial].
            Default: `0.0` (no rotation).
        translation: See [`Spatial`][torchio.Spatial].
        isotropic: See [`Spatial`][torchio.Spatial].
        center: See [`Spatial`][torchio.Spatial].
        default_pad_value: See [`Spatial`][torchio.Spatial].
        default_pad_label: See [`Spatial`][torchio.Spatial].
        image_interpolation: See [`Spatial`][torchio.Spatial].
        label_interpolation: See [`Spatial`][torchio.Spatial].
        one_hot_label_interpolation: See [`Spatial`][torchio.Spatial].
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Affine(degrees=(-15, 15))
        >>> transform = tio.Affine(scales=1.0, degrees=(0, 0, 90))
    """

    def __init__(
        self,
        *,
        scales: TypeParameterValue = 1.0,
        degrees: TypeParameterValue = 0.0,
        translation: TypeParameterValue = 0.0,
        isotropic: bool = False,
        center: TypeCenter = "image",
        default_pad_value: TypePadValue | float = "minimum",
        default_pad_label: int | float = 0,
        image_interpolation: TypeImageInterpolation | int = "linear",
        label_interpolation: TypeLabelInterpolation | int = "nearest",
        one_hot_label_interpolation: TypeImageInterpolation | int = "linear",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            scales=scales,
            degrees=degrees,
            translation=translation,
            isotropic=isotropic,
            center=center,
            default_pad_value=default_pad_value,
            default_pad_label=default_pad_label,
            image_interpolation=image_interpolation,
            label_interpolation=label_interpolation,
            one_hot_label_interpolation=one_hot_label_interpolation,
            **kwargs,
        )
        self._warn_if_noop(
            is_noop=(
                self.scales.is_constant(1.0)
                and self.degrees.is_constant(0.0)
                and self.translation.is_constant(0.0)
            ),
            hint="degrees=(-15, 15)",
        )


class ElasticDeformation(Spatial):
    r"""Apply a dense random elastic deformation.

    Convenience wrapper around [`Spatial`][torchio.Spatial] exposing
    only the elastic parameters.

    A random displacement is assigned to a coarse grid of control
    points and trilinearly upsampled to the image resolution.

    Args:
        control_points: See [`Spatial`][torchio.Spatial].
        num_control_points: See [`Spatial`][torchio.Spatial].
        max_displacement: See [`Spatial`][torchio.Spatial].
            Default: `7.5` mm.
        locked_borders: See [`Spatial`][torchio.Spatial].
        image_interpolation: See [`Spatial`][torchio.Spatial].
        label_interpolation: See [`Spatial`][torchio.Spatial].
        one_hot_label_interpolation: See [`Spatial`][torchio.Spatial].
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.ElasticDeformation()
        >>> transform = tio.ElasticDeformation(
        ...     max_displacement=10,
        ...     num_control_points=5,
        ... )
    """

    def __init__(
        self,
        *,
        control_points: TypeControlPoints | None = None,
        num_control_points: int | TypeThreeInts = 7,
        max_displacement: TypeParameterValue = 7.5,
        locked_borders: int = 2,
        image_interpolation: TypeImageInterpolation | int = "linear",
        label_interpolation: TypeLabelInterpolation | int = "nearest",
        one_hot_label_interpolation: TypeImageInterpolation | int = "linear",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            control_points=control_points,
            num_control_points=num_control_points,
            max_displacement=max_displacement,
            locked_borders=locked_borders,
            image_interpolation=image_interpolation,
            label_interpolation=label_interpolation,
            one_hot_label_interpolation=one_hot_label_interpolation,
            **kwargs,
        )


def _invert_per_sample(params: dict[str, Any]) -> _PerSampleGrids:
    """Build per-element inverse grid inputs from per-instance params.

    Affine matrices are inverted exactly and elastic fields are negated,
    matching the batch-shared inverse. Identity (`None`) elements stay
    identity.

    Args:
        params: The per-instance parameter dict from `make_params`.

    Returns:
        A `_PerSampleGrids` with inverted per-element parameters.
    """
    inverse_affines: list[np.ndarray | None] = []
    inverse_control_points: list[Tensor | None] = []
    inverse_displacements: list[tuple[float, float, float] | None] = []
    for matrix, field in zip(
        params["affine_matrix"],
        params["control_points"],
        strict=True,
    ):
        affine = _deserialize_matrix(matrix)
        inverse_affines.append(np.linalg.inv(affine) if affine is not None else None)
        control_points = _deserialize_control_points(field)
        if control_points is not None:
            control_points = -control_points
            inverse_displacements.append(_max_abs_displacement(control_points))
        else:
            inverse_displacements.append(None)
        inverse_control_points.append(control_points)
    return _PerSampleGrids(
        inverse_affines,
        inverse_control_points,
        inverse_displacements,
    )


def _resolve_spatial_params(
    params: dict[str, Any],
) -> tuple[
    np.ndarray | None,
    Tensor | None,
    tuple[float, float, float] | None,
    _PerSampleGrids | None,
]:
    """Deserialize spatial params into shared or per-sample grid inputs.

    Returns:
        A tuple `(affine_matrix, control_points, max_displacement,
        per_sample)`. For batch-shared params the first three are
        populated and `per_sample` is `None`. For per-instance params
        with at least one geometric element, only `per_sample` is
        populated. Per-instance params with no geometry collapse to the
        shared no-op path (all `None`).
    """
    batched_keys = params.get("_batched_keys") or []
    if "affine_matrix" not in batched_keys:
        affine_matrix = _deserialize_matrix(params["affine_matrix"])
        control_points = _deserialize_control_points(params["control_points"])
        max_displacement = _deserialize_max_displacement(params["max_displacement"])
        return affine_matrix, control_points, max_displacement, None

    affine_matrices = [_deserialize_matrix(m) for m in params["affine_matrix"]]
    control_points_list = [
        _deserialize_control_points(c) for c in params["control_points"]
    ]
    displacements = [
        _deserialize_max_displacement(d) for d in params["max_displacement"]
    ]
    has_geometry = any(m is not None for m in affine_matrices) or any(
        c is not None for c in control_points_list
    )
    if not has_geometry:
        return None, None, None, None
    return (
        None,
        None,
        None,
        _PerSampleGrids(
            affine_matrices,
            control_points_list,
            displacements,
        ),
    )


def _build_grid_per_sample_checked(
    *,
    per_sample: _PerSampleGrids,
    first_img_batch: ImagesBatch,
    input_shape: TypeThreeInts,
    input_affine: AffineMatrix,
    output_shape: TypeThreeInts,
    output_affine: AffineMatrix,
    affine_first: bool,
) -> Tensor:
    """Build per-element sampling grids after validating the batch size.

    Args:
        per_sample: The per-element geometry recorded at sampling time.
        first_img_batch: First selected image batch (provides device and
            batch size).
        input_shape: Spatial shape of the input grid.
        input_affine: Affine of the input grid.
        output_shape: Spatial shape of the output grid.
        output_affine: Affine of the output grid.
        affine_first: Whether the affine is applied before the elastic
            displacement.

    Returns:
        The stacked per-element sampling grid.

    Raises:
        RuntimeError: If the number of recorded elements does not match
            the batch size.
    """
    if len(per_sample.affine_matrices) != first_img_batch.batch_size:
        msg = (
            "Per-instance spatial parameters were recorded for"
            f" {len(per_sample.affine_matrices)} elements but the batch"
            f" has {first_img_batch.batch_size}"
        )
        raise RuntimeError(msg)
    return _build_per_sample_grids(
        input_shape=input_shape,
        input_affine=input_affine,
        output_shape=output_shape,
        output_affine=output_affine,
        affine_matrices=per_sample.affine_matrices,
        control_points_list=per_sample.control_points,
        max_displacements=per_sample.max_displacements,
        affine_first=affine_first,
        device=first_img_batch.data.device,
    )


def _passthrough_indices(per_sample: _PerSampleGrids) -> list[int]:
    """Indices of batch elements with no geometry (identity / gated out).

    Args:
        per_sample: The per-element geometry recorded at sampling time.

    Returns:
        Indices whose affine and control points are both ``None``, i.e.
        elements that should pass through a no-target resample unchanged.
    """
    return [
        index
        for index, affine in enumerate(per_sample.affine_matrices)
        if affine is None and per_sample.control_points[index] is None
    ]


def _restore_spatial_output_affines(
    img_batch: ImagesBatch,
    *,
    original_data: Tensor,
    original_affines: list[AffineMatrix],
    output_affine: AffineMatrix,
    passthrough: list[int],
) -> None:
    """Set the output affines and restore no-op (passthrough) elements.

    Every resampled element adopts *output_affine*. Passthrough elements
    (no geometry, no target) instead keep their original data and affine
    so that per-element gating is an exact no-op for them, undoing the
    tiny perturbation introduced by the float32 identity resample.

    Args:
        img_batch: The image batch whose data/affines are updated in place.
        original_data: The input data, before resampling.
        original_affines: The input affines, before resampling.
        output_affine: The affine shared by every resampled element.
        passthrough: Indices of elements to restore exactly.
    """
    new_affines = [output_affine.clone() for _ in img_batch.affines]
    if passthrough:
        data = img_batch.data
        for index in passthrough:
            data[index] = original_data[index]
            new_affines[index] = original_affines[index]
        img_batch.data = data
    img_batch.affines[:] = new_affines


def _apply_spatial_to_batch(
    *,
    batch: SubjectsBatch,
    image_names: list[str],
    target_space: TypeTargetSpace | None,
    affine_matrix: np.ndarray | None,
    control_points: Tensor | None,
    max_displacement: tuple[float, float, float] | None,
    affine_first: bool,
    image_interpolation: TypeImageInterpolation,
    label_interpolation: TypeLabelInterpolation,
    one_hot_label_interpolation: TypeImageInterpolation = "linear",
    antialias: bool,
    default_pad_value: TypePadValue | float,
    default_pad_label: float,
    per_sample: _PerSampleGrids | None = None,
) -> None:
    """Apply the spatial mapping to all selected images in *batch*.

    By default a single sampling grid is built from the first image's
    geometry and reused for every image and every sample in the batch.
    When *per_sample* is given, one grid is built per batch element so
    each element is augmented independently. Only the interpolation mode
    and fill value change per image type.
    """
    if not image_names:
        return

    first_img_batch = batch.images[image_names[0]]
    input_shape = _get_spatial_shape(first_img_batch)
    input_affine = first_img_batch.affines[0]
    output_shape = target_space[0] if target_space is not None else input_shape
    output_affine = target_space[1] if target_space is not None else input_affine

    if per_sample is None:
        grid = _build_sampling_grid(
            input_shape=input_shape,
            input_affine=input_affine,
            output_shape=output_shape,
            output_affine=output_affine,
            affine_matrix=affine_matrix,
            control_points=control_points,
            max_displacement=max_displacement,
            affine_first=affine_first,
            device=first_img_batch.data.device,
        )
    else:
        grid = _build_grid_per_sample_checked(
            per_sample=per_sample,
            first_img_batch=first_img_batch,
            input_shape=input_shape,
            input_affine=input_affine,
            output_shape=output_shape,
            output_affine=output_affine,
            affine_first=affine_first,
        )

    # Elements with no geometry and no target resampling must be exact
    # no-ops, but the batched per-sample resample computes through an
    # identity grid in float32, so restore those rows afterwards.
    passthrough = (
        _passthrough_indices(per_sample)
        if per_sample is not None and target_space is None
        else []
    )

    for name in image_names:
        _resample_image_batch(
            batch.images[name],
            grid=grid,
            per_sample=per_sample,
            input_shape=input_shape,
            input_affine=input_affine,
            output_affine=output_affine,
            image_interpolation=image_interpolation,
            label_interpolation=label_interpolation,
            one_hot_label_interpolation=one_hot_label_interpolation,
            antialias=antialias,
            default_pad_value=default_pad_value,
            default_pad_label=default_pad_label,
            passthrough=passthrough,
        )


def _resample_image_batch(
    img_batch: ImagesBatch,
    *,
    grid: Tensor,
    per_sample: _PerSampleGrids | None,
    input_shape: TypeThreeInts,
    input_affine: AffineMatrix,
    output_affine: AffineMatrix,
    image_interpolation: TypeImageInterpolation,
    label_interpolation: TypeLabelInterpolation,
    one_hot_label_interpolation: TypeImageInterpolation = "linear",
    antialias: bool,
    default_pad_value: TypePadValue | float,
    default_pad_label: float,
    passthrough: list[int],
) -> None:
    """Resample a single image batch and restore no-op elements.

    Args:
        img_batch: The image batch to resample in place.
        grid: The shared or per-sample sampling grid.
        per_sample: Per-element geometry, or ``None`` for the shared grid.
        input_shape: Spatial shape of the input volume.
        input_affine: Affine of the input volume.
        output_affine: Affine shared by every resampled element.
        image_interpolation: Interpolation mode for scalar images.
        label_interpolation: Interpolation mode for label maps.
        one_hot_label_interpolation: Per-channel interpolation used by the
            ``"label"`` partial-volume mode.
        antialias: Whether to blur scalar images before downsampling.
        default_pad_value: Fill value for out-of-bounds scalar samples.
        default_pad_label: Fill value for out-of-bounds label samples.
        passthrough: Indices of no-op elements to restore exactly.
    """
    original_data = img_batch.data
    original_affines = list(img_batch.affines)
    is_label = issubclass(img_batch._image_class, LabelMap)
    interpolation = _interpolation_for_batch(
        img_batch,
        image_interpolation=image_interpolation,
        label_interpolation=label_interpolation,
    )
    if is_label and interpolation == LABEL_INTERPOLATION:
        img_batch.data = _resample_label_partial_volume(
            img_batch.data,
            grid,
            input_shape=input_shape,
            input_affine=input_affine,
            output_affine=output_affine,
            antialias=antialias,
            one_hot_label_interpolation=one_hot_label_interpolation,
            per_sample=per_sample is not None,
            default_pad_label=float(default_pad_label),
        )
    else:
        fill_value = _batch_fill_value(
            img_batch,
            default_pad_value=default_pad_value,
            default_pad_label=default_pad_label,
        )
        data = img_batch.data
        # Antialias: blur ScalarImages before downsampling.
        if antialias and not is_label:
            data = _antialias_batch(data, input_affine, output_affine)
        sampler = _sample_batch if per_sample is None else _sample_batch_per_sample
        img_batch.data = sampler(
            data,
            grid,
            input_shape=input_shape,
            interpolation=interpolation,
            fill_value=fill_value,
        )
    _restore_spatial_output_affines(
        img_batch,
        original_data=original_data,
        original_affines=original_affines,
        output_affine=output_affine,
        passthrough=passthrough,
    )


def _resample_label_partial_volume(
    data: Tensor,
    grid: Tensor,
    *,
    input_shape: TypeThreeInts,
    input_affine: AffineMatrix,
    output_affine: AffineMatrix,
    antialias: bool,
    one_hot_label_interpolation: TypeImageInterpolation = "linear",
    per_sample: bool = False,
    default_pad_label: float,
) -> Tensor:
    r"""Resample a discrete label map in a partial-volume-aware way.

    Nearest-neighbor resampling of a label map produces staircase artifacts
    and can drop thin structures, especially when downsampling. This helper
    instead:

    1. one-hot encodes the label map using the unique label values present
       (robust to non-contiguous labels such as $\{0, 2, 5\}$),
    2. optionally Gaussian-smooths each channel before downsampling when
       *antialias* is `True` (using the same `_antialias_sigmas` as the
       intensity antialiasing path, from Cardoso et al., MICCAI 2015),
    3. resamples every channel with *one_hot_label_interpolation*
       (`"linear"` by default; higher-order B-spline modes give smoother
       boundaries, which is useful when upsampling), and
    4. takes the per-voxel argmax to recover the discrete labels.

    This single-channel pipeline (`C == 1`) fills out-of-bounds voxels with
    *default_pad_label*.

    Multi-channel inputs (`C > 1`, an already one-hot or probabilistic map)
    take a different path: their channels are resampled with
    *one_hot_label_interpolation* **without** re-encoding or an argmax,
    out-of-bounds voxels are filled with `0` (*default_pad_label* is **not**
    applied), and the partial-volume result is returned as floating point so
    the interpolated fractions are not truncated.

    Args:
        data: `(B, C, I, J, K)` label batch. When `C == 1` the full one-hot
            pipeline is used. When `C > 1` the channels are resampled
            without re-encoding (see above).
        grid: `(I_out, J_out, K_out, 3)` sampling grid in input voxel
            coordinates.
        input_shape: Spatial shape of the input volume `(I, J, K)`.
        input_affine: Affine of the input grid.
        output_affine: Affine of the output grid.
        antialias: Whether to Gaussian-smooth the one-hot channels before
            downsampling.
        one_hot_label_interpolation: Interpolation used to resample the
            one-hot channels (`"nearest"`, `"linear"`, or a higher-order
            B-spline mode). Defaults to `"linear"`.
        per_sample: If `True`, *grid* carries a batch dimension
            `(B, I_out, J_out, K_out, 3)` and each element is resampled with
            its own coordinates (per-instance augmentation); otherwise a
            single shared grid is used.
        default_pad_label: Value assigned to out-of-bounds voxels. Only
            applied for single-channel (`C == 1`) inputs; multi-channel
            inputs use `0` for out-of-bounds.

    Returns:
        Resampled `(B, 1, I_out, J_out, K_out)` label batch with the input
        dtype when `C == 1`. When `C > 1`, a `(B, C, ...)` partial-volume
        batch is returned: the input dtype is preserved for floating-point
        inputs, otherwise the output is `float32` to avoid truncating the
        interpolated values.
    """
    sampler = _sample_batch_per_sample if per_sample else _sample_batch
    if data.shape[1] > 1:
        smoothed = data.float()
        if antialias:
            smoothed = _antialias_batch(smoothed, input_affine, output_affine)
        sampled = sampler(
            smoothed,
            grid,
            input_shape=input_shape,
            interpolation=one_hot_label_interpolation,
            fill_value=0.0,
        )
        if data.dtype.is_floating_point:
            return sampled.to(data.dtype)
        return sampled

    labels = torch.unique(data)
    values = rearrange(data[:, 0], "b i j k -> b 1 i j k")
    targets = rearrange(labels, "n -> 1 n 1 1 1")
    one_hot = (values == targets).float()

    if antialias:
        one_hot = _antialias_batch(one_hot, input_affine, output_affine)

    sampled = sampler(
        one_hot,
        grid,
        input_shape=input_shape,
        interpolation=one_hot_label_interpolation,
        fill_value=0.0,
    )

    winners = sampled.argmax(dim=1)
    resampled = labels[winners]
    # In-bounds voxels keep partition of unity (channels sum to ~1). Near the
    # border, the channel sum equals the in-bounds fraction of the sampling
    # neighborhood. A voxel sampled mostly (>50%) from outside the input is
    # treated as out-of-bounds and set to default_pad_label, matching the
    # `mask > 0.5` fill convention used for intensity images in
    # `_sample_batch_grid_sample`.
    in_bounds = sampled.sum(dim=1) > 0.5
    resampled = torch.where(
        in_bounds,
        resampled,
        torch.full_like(resampled, default_pad_label),
    )
    out = rearrange(resampled, "b i j k -> b 1 i j k")
    return out.to(data.dtype)


def _resolve_target_space(
    target: TypeTarget,
    batch: SubjectsBatch,
    first_shape: TypeThreeInts,
    first_affine: AffineMatrix,
) -> TypeTargetSpace | None:
    """Convert the user-facing *target* specification to `(shape, affine)`.

    Accepts scalars (isotropic spacing), 3-tuples (per-axis spacing), image
    names in the subject, file paths, `Image` instances, or explicit
    `(shape, affine)` pairs.  Returns `None` when the output grid should
    match the input grid.
    """
    if target is None:
        return None
    if isinstance(target, Image):
        return target.spatial_shape, target.affine.clone()
    if isinstance(target, (str, Path)):
        return _target_from_string_or_path(target, batch, first_shape, first_affine)
    if _is_target_space_tuple(target):
        shape, affine = target
        return _parse_target_space_tuple(shape, affine)
    # Remaining cases: int, float, tuple of numbers, list, ndarray, or a
    # random spacing spec (range, Choice, or Distribution).
    if not isinstance(
        target, (int, float, tuple, list, np.ndarray, Choice, Distribution)
    ):
        msg = f'Target not understood: "{target}"'
        raise ValueError(msg)
    spacing = _resolve_target_spacing(target)
    return _compute_new_shape_affine(first_shape, first_affine, spacing)


def _target_from_string_or_path(
    target: str | Path,
    batch: SubjectsBatch,
    first_shape: TypeThreeInts,
    first_affine: AffineMatrix,
) -> TypeTargetSpace:
    """Resolve a string or Path target (file, image name, or spacing)."""
    path = Path(target)
    if path.is_file():
        image = ScalarImage(path)
        return image.spatial_shape, image.affine.clone()
    if isinstance(target, str) and target in batch.images:
        reference = batch.images[target]
        return _get_spatial_shape(reference), reference.affines[0].clone()
    msg = (
        f'Unknown target "{target}". Pass a file path, an image name in the'
        " subject, an Image, or a spacing specification"
    )
    raise ValueError(msg)


def _resolve_target_spacing(
    value: TypeSpacingSpec,
) -> TypeSpacing:
    """Resolve a (possibly random) spacing spec to a positive 3-tuple.

    Scalars and 3-number sequences are deterministic. Ranges (a 2-tuple
    `(lo, hi)` or a 6-tuple of per-axis ranges), a `Choice`, or a
    `torch.distributions.Distribution` are sampled once, here, at apply time,
    using the same convention as the other spatial parameters. NumPy arrays are
    treated as deterministic 3-element spacings.

    Args:
        value: A scalar, sequence, range, `Choice`, or `Distribution`.

    Returns:
        A strictly-positive `(sx, sy, sz)` spacing tuple.
    """
    if isinstance(value, np.ndarray):
        return _parse_spacing(tuple(float(v) for v in value.flat))
    if isinstance(value, (int, float)):
        return _parse_spacing(float(value))
    spec = tuple(value) if isinstance(value, list) else value
    sampled = _ParameterRange(spec).sample()
    return _parse_spacing(sampled)


def _compute_new_shape_affine(
    shape: TypeThreeInts,
    affine: AffineMatrix,
    spacing: TypeSpacing,
) -> TypeTargetSpace:
    """Compute the output shape and affine for a target voxel spacing.

    The output grid is centered on the same physical center as the input
    grid.  Axes with a single voxel (size 1) are left unchanged.
    """
    old_spacing = np.asarray(affine.spacing, dtype=np.float64)
    new_spacing = np.asarray(spacing, dtype=np.float64)
    old_shape = np.asarray(shape, dtype=np.float64)

    new_shape = np.floor(old_shape * old_spacing / new_spacing)
    new_shape[old_shape == 1] = 1

    # Keep the physical center of the volume fixed when changing spacing.
    rotation = affine.direction.cpu().numpy()
    old_origin = np.asarray(affine.origin, dtype=np.float64)
    old_center = old_origin + rotation @ (((old_shape - 1) / 2) * old_spacing)
    new_origin = old_center - rotation @ (((new_shape - 1) / 2) * new_spacing)

    new_affine = np.eye(4, dtype=np.float64)
    new_affine[:3, :3] = rotation * new_spacing
    new_affine[:3, 3] = new_origin
    return (
        (int(new_shape[0]), int(new_shape[1]), int(new_shape[2])),
        AffineMatrix(new_affine),
    )


def _build_sampling_grid(
    *,
    input_shape: TypeThreeInts,
    input_affine: AffineMatrix,
    output_shape: TypeThreeInts,
    output_affine: AffineMatrix,
    affine_matrix: np.ndarray | None,
    control_points: Tensor | None,
    max_displacement: tuple[float, float, float] | None,
    affine_first: bool,
    device: torch.device,
) -> Tensor:
    """Build a sampling grid in **input voxel coordinates**.

    The grid maps each output voxel to its source location in the input
    volume.  The mapping is:

    .. code-block:: text

        output voxel  →  world  →  (optional inverse affine)  →  input voxel

    When elastic control points are provided, the dense displacement
    field (in mm) is converted to voxel offsets and added to the mapped
    coordinates.  The `affine_first` flag controls whether the affine
    mapping or the elastic displacement is applied first to the image.

    Because the grid pulls values from the input, the operation applied
    first to the image is the one applied last to the coordinates.  The
    elastic field spans the output grid, and both operations act in the
    output space.  With output voxel *x*, affine *A* and displacement *d*,
    `affine_first=True` samples the input at `A⁻¹(x + d(x))` and
    `affine_first=False` samples it at `y + d(y)` with `y = A⁻¹(x)`,
    matching sequential application of [`Affine`][torchio.Affine] and
    [`ElasticDeformation`][torchio.ElasticDeformation] in each order after
    resampling to the output grid.

    Returns:
        Grid tensor with shape `(I_out, J_out, K_out, 3)` in input
        voxel coordinates.
    """
    mapping = _output_to_input_voxel_matrix(
        input_affine=input_affine,
        output_affine=output_affine,
        affine_matrix=affine_matrix,
        device=device,
    )
    output_coords = _output_voxel_coordinates(output_shape, device)

    if control_points is None:
        return _apply_voxel_mapping(output_coords, mapping)

    output_spacing = np.asarray(output_affine.spacing, dtype=np.float64)
    if max_displacement is None:
        max_displacement = _max_abs_displacement(control_points)
    _check_folding(
        control_points.cpu().numpy(),
        max_displacement,
        output_shape,
        output_spacing,
    )
    control_points = control_points.to(device=device, dtype=torch.float32)
    # The elastic field spans the output grid in both orders.
    output_spacing_t = torch.as_tensor(
        output_affine.spacing,
        dtype=torch.float32,
        device=device,
    )

    if affine_first:
        # Affine first, then elastic: deform the output coordinates and
        # then map them to the input.
        displacement = _upsample_displacement_field(control_points, output_shape)
        deformed_output = output_coords + displacement / output_spacing_t
        input_voxels = _apply_voxel_mapping(deformed_output, mapping)
    else:
        # Elastic first, then affine: apply the inverse affine within the
        # output grid, add the displacement at that location, and only
        # then map to the input.  The control points are interpolated
        # directly at the moved coordinates, which agrees with
        # `_upsample_displacement_field` on the grid nodes.
        affine_in_output = _output_to_input_voxel_matrix(
            input_affine=output_affine,
            output_affine=output_affine,
            affine_matrix=affine_matrix,
            device=device,
        )
        output_to_input = _output_to_input_voxel_matrix(
            input_affine=input_affine,
            output_affine=output_affine,
            affine_matrix=None,
            device=device,
        )
        moved_output = _apply_voxel_mapping(output_coords, affine_in_output)
        control_shape = control_points.shape[:3]
        to_control = torch.tensor(
            [
                (n - 1) / max(s - 1, 1)
                for n, s in zip(control_shape, output_shape, strict=True)
            ],
            dtype=torch.float32,
            device=device,
        )
        displacement = _sample_batch_grid_sample(
            rearrange(control_points, "i j k d -> 1 d i j k"),
            moved_output * to_control,
            input_shape=(control_shape[0], control_shape[1], control_shape[2]),
            mode="bilinear",
            fill_value=0.0,
        )
        displacement = rearrange(displacement, "1 d i j k -> i j k d")
        deformed_output = moved_output + displacement / output_spacing_t
        input_voxels = _apply_voxel_mapping(deformed_output, output_to_input)

    return input_voxels


def _output_to_input_voxel_matrix(
    *,
    input_affine: AffineMatrix,
    output_affine: AffineMatrix,
    affine_matrix: np.ndarray | None,
    device: torch.device,
) -> Tensor:
    """Compute the 4x4 matrix mapping output voxels to input voxels.

    The composition is `A_in^{-1} @ T^{-1} @ A_out` where *T* is the
    optional world-space affine transform.
    """
    input_affine_inv = np.linalg.inv(input_affine.numpy())
    transform_inv = (
        np.eye(4, dtype=np.float64)
        if affine_matrix is None
        else np.linalg.inv(np.asarray(affine_matrix, dtype=np.float64))
    )
    matrix = input_affine_inv @ transform_inv @ output_affine.numpy()
    return torch.as_tensor(matrix, dtype=torch.float32, device=device)


def _output_voxel_coordinates(
    shape: TypeThreeInts,
    device: torch.device,
) -> Tensor:
    """Create an `(I, J, K, 3)` meshgrid of output voxel indices."""
    i = torch.arange(shape[0], dtype=torch.float32, device=device)
    j = torch.arange(shape[1], dtype=torch.float32, device=device)
    k = torch.arange(shape[2], dtype=torch.float32, device=device)
    grid_i, grid_j, grid_k = torch.meshgrid(i, j, k, indexing="ij")
    return torch.stack([grid_i, grid_j, grid_k], dim=-1)


def _apply_voxel_mapping(
    coords: Tensor,
    matrix: Tensor,
) -> Tensor:
    """Apply a 4x4 homogeneous matrix to an `(..., 3)` coordinate tensor."""
    ones = torch.ones(*coords.shape[:-1], 1, dtype=coords.dtype, device=coords.device)
    homogeneous = torch.cat([coords, ones], dim=-1)
    mapped = homogeneous @ matrix.T
    return mapped[..., :3]


def _voxel_coordinates_to_grid(
    coords: Tensor,
    input_shape: TypeThreeInts,
) -> Tensor:
    """Normalize `(I, J, K, 3)` voxel coords to the `[-1, 1]` grid.

    `F.grid_sample` expects coordinates in `[-1, 1]` where `-1`
    maps to the first voxel and `+1` to the last.  The output is
    rearranged to `(1, K, J, I, 3)` to match PyTorch's `(D, H, W)`
    convention.
    """
    size_i = max(input_shape[0] - 1, 1)
    size_j = max(input_shape[1] - 1, 1)
    size_k = max(input_shape[2] - 1, 1)
    sizes = torch.tensor(
        [size_i, size_j, size_k],
        dtype=torch.float32,
        device=coords.device,
    )
    grid = 2.0 * coords / sizes - 1.0
    # (I, J, K, 3) -> (1, K, J, I, 3) for grid_sample
    return rearrange(grid, "i j k d -> 1 k j i d")


def _sample_batch(
    data: Tensor,
    voxel_grid: Tensor,
    *,
    input_shape: TypeThreeInts,
    interpolation: str,
    fill_value: float | Tensor,
) -> Tensor:
    """Resample a 5D batch using a shared sampling grid.

    For interpolation orders 0-1 (nearest, linear), the fast
    `F.grid_sample` path is used.  For orders 2+ (quadratic,
    cubic, ...), `interpol.grid_pull` provides high-order B-spline
    interpolation.

    Args:
        data: `(B, C, I, J, K)` image batch.
        voxel_grid: `(I_out, J_out, K_out, 3)` sampling grid in
            input voxel coordinates.
        input_shape: Spatial shape of the input volume `(I, J, K)`.
        interpolation: Interpolation mode name.
        fill_value: Scalar or per-channel fill for out-of-bounds
            samples.

    Returns:
        Resampled `(B, C, I_out, J_out, K_out)` tensor.
    """
    order = _INTERPOLATION_TO_ORDER[interpolation]
    if order <= 1:
        return _sample_batch_grid_sample(
            data,
            voxel_grid,
            input_shape=input_shape,
            mode=_TORCH_INTERPOLATION_MODE[interpolation],
            fill_value=fill_value,
        )
    return _sample_batch_interpol(
        data,
        voxel_grid,
        order=order,
        fill_value=fill_value,
    )


def _sample_batch_grid_sample(
    data: Tensor,
    voxel_grid: Tensor,
    *,
    input_shape: TypeThreeInts,
    mode: str,
    fill_value: float | Tensor,
) -> Tensor:
    """Fast path: resample with F.grid_sample (orders 0-1)."""
    batch_size = data.shape[0]
    # Normalize voxel coords to [-1, 1] for grid_sample.
    grid = _voxel_coordinates_to_grid(voxel_grid, input_shape)
    # (B, C, I, J, K) -> (B, C, K, J, I) for grid_sample
    input_5d = rearrange(data, "b c i j k -> b c k j i").float()
    # Expand grid from (1, ...) to (B, ...)
    grid_b = grid.expand(batch_size, -1, -1, -1, -1)
    sampled = functional.grid_sample(
        input_5d,
        grid_b,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )

    fill_tensor = _prepare_fill_value(fill_value, input_5d)
    if fill_tensor is not None:
        ones = torch.ones_like(input_5d)
        mask = functional.grid_sample(
            ones,
            grid_b,
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = torch.where(mask > 0.5, sampled, fill_tensor)

    # (B, C, K, J, I) -> (B, C, I, J, K)
    return rearrange(sampled, "b c k j i -> b c i j k").to(data.dtype)


def _sample_batch_interpol(
    data: Tensor,
    voxel_grid: Tensor,
    *,
    order: int,
    fill_value: float | Tensor,
) -> Tensor:
    """High-quality path: resample with interpol.grid_pull (orders 2+).

    `interpol.grid_pull` works in voxel coordinates natively and
    supports B-spline orders up to 7.
    """
    import interpol

    batch_size = data.shape[0]
    # interpol expects: input (B, C, *spatial), grid (B, *spatial, D)
    # Our grid is (I_out, J_out, K_out, 3). Add batch dim.
    grid_b = voxel_grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

    sampled = interpol.grid_pull(
        data.float(),
        grid_b,
        interpolation=order,
        bound="dct2",
        extrapolate=False,
        prefilter=True,
    )
    return sampled.to(data.dtype)


def _sample_batch_per_sample(
    data: Tensor,
    voxel_grid: Tensor,
    *,
    input_shape: TypeThreeInts,
    interpolation: str,
    fill_value: float | Tensor,
) -> Tensor:
    """Resample a 5D batch using a per-sample sampling grid.

    Unlike [`_sample_batch`][], the grid carries its own batch dimension
    so each element of the batch is resampled with independent
    coordinates (per-instance augmentation).

    Args:
        data: `(B, C, I, J, K)` image batch.
        voxel_grid: `(B, I_out, J_out, K_out, 3)` per-sample sampling
            grid in input voxel coordinates.
        input_shape: Spatial shape of the input volume `(I, J, K)`.
        interpolation: Interpolation mode name.
        fill_value: Scalar or per-channel fill for out-of-bounds samples.

    Returns:
        Resampled `(B, C, I_out, J_out, K_out)` tensor.
    """
    order = _INTERPOLATION_TO_ORDER[interpolation]
    if order <= 1:
        return _sample_batch_grid_sample_per_sample(
            data,
            voxel_grid,
            input_shape=input_shape,
            mode=_TORCH_INTERPOLATION_MODE[interpolation],
            fill_value=fill_value,
        )
    return _sample_batch_interpol_per_sample(
        data,
        voxel_grid,
        order=order,
        fill_value=fill_value,
    )


def _voxel_coordinates_to_grid_batched(
    coords: Tensor,
    input_shape: TypeThreeInts,
) -> Tensor:
    """Normalize `(B, I, J, K, 3)` voxel coords to the `[-1, 1]` grid.

    Like [`_voxel_coordinates_to_grid`][] but keeps a leading batch
    dimension, producing `(B, K, J, I, 3)` for `F.grid_sample`.
    """
    size_i = max(input_shape[0] - 1, 1)
    size_j = max(input_shape[1] - 1, 1)
    size_k = max(input_shape[2] - 1, 1)
    sizes = torch.tensor(
        [size_i, size_j, size_k],
        dtype=torch.float32,
        device=coords.device,
    )
    grid = 2.0 * coords / sizes - 1.0
    return rearrange(grid, "b i j k d -> b k j i d")


def _sample_batch_grid_sample_per_sample(
    data: Tensor,
    voxel_grid: Tensor,
    *,
    input_shape: TypeThreeInts,
    mode: str,
    fill_value: float | Tensor,
) -> Tensor:
    """Per-sample fast path: resample with F.grid_sample (orders 0-1)."""
    grid = _voxel_coordinates_to_grid_batched(voxel_grid, input_shape)
    input_5d = rearrange(data, "b c i j k -> b c k j i").float()
    sampled = functional.grid_sample(
        input_5d,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )

    fill_tensor = _prepare_fill_value(fill_value, input_5d)
    if fill_tensor is not None:
        ones = torch.ones_like(input_5d)
        mask = functional.grid_sample(
            ones,
            grid,
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = torch.where(mask > 0.5, sampled, fill_tensor)

    return rearrange(sampled, "b c k j i -> b c i j k").to(data.dtype)


def _sample_batch_interpol_per_sample(
    data: Tensor,
    voxel_grid: Tensor,
    *,
    order: int,
    fill_value: float | Tensor,
) -> Tensor:
    """Per-sample high-quality path: resample with interpol.grid_pull (orders 2+)."""
    import interpol

    sampled = interpol.grid_pull(
        data.float(),
        voxel_grid,
        interpolation=order,
        bound="dct2",
        extrapolate=False,
        prefilter=True,
    )
    return sampled.to(data.dtype)


def _build_per_sample_grids(
    *,
    input_shape: TypeThreeInts,
    input_affine: AffineMatrix,
    output_shape: TypeThreeInts,
    output_affine: AffineMatrix,
    affine_matrices: list[np.ndarray | None],
    control_points_list: list[Tensor | None],
    max_displacements: list[tuple[float, float, float] | None],
    affine_first: bool,
    device: torch.device,
) -> Tensor:
    """Build one sampling grid per batch element and stack them.

    Each element reuses the shared single-grid builder with its own
    affine matrix and elastic field, so the per-instance path inherits
    the exact geometry of the batch-shared path. A `None` affine and
    `None` field yield an identity grid (used for gated-out elements).

    Returns:
        A `(B, I_out, J_out, K_out, 3)` tensor of input voxel
        coordinates.
    """
    grids = [
        _build_sampling_grid(
            input_shape=input_shape,
            input_affine=input_affine,
            output_shape=output_shape,
            output_affine=output_affine,
            affine_matrix=affine_matrices[index],
            control_points=control_points_list[index],
            max_displacement=max_displacements[index],
            affine_first=affine_first,
            device=device,
        )
        for index in range(len(affine_matrices))
    ]
    return torch.stack(grids, dim=0)


def _antialias_batch(
    data: Tensor,
    input_affine: AffineMatrix,
    output_affine: AffineMatrix,
) -> Tensor:
    """Apply Gaussian smoothing before downsampling.

    Only axes whose output spacing is larger than the input spacing
    (i.e., axes being downsampled) are smoothed. The standard deviations
    follow Cardoso et al., `Scale factor point spread function matching
    <https://link.springer.com/chapter/10.1007/978-3-319-24571-3_81>`_,
    MICCAI 2015.

    Args:
        data: `(B, C, I, J, K)` image batch.
        input_affine: Affine of the input grid.
        output_affine: Affine of the output grid.

    Returns:
        Smoothed `(B, C, I, J, K)` tensor.
    """
    input_spacing = np.asarray(input_affine.spacing, dtype=np.float64)
    output_spacing = np.asarray(output_affine.spacing, dtype=np.float64)
    factors = output_spacing / input_spacing
    sigmas = _antialias_sigmas(factors, input_spacing)
    if np.all(sigmas == 0):
        return data
    return _gaussian_smooth_batch(data, sigmas)


def _antialias_sigmas(
    factors: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """Compute per-axis Gaussian sigma for antialiasing.

    From Cardoso et al., MICCAI 2015, Eq. at top of p. 678:
    `variance = (k^2 - 1) / (2 * sqrt(2 * ln(2)))^2`

    Args:
        factors: Per-axis downsampling factor (output / input spacing).
        spacing: Input voxel spacing in mm.

    Returns:
        Per-axis sigma in voxels. Zero for axes not being downsampled.
    """
    sigmas = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        k = factors[axis]
        if k <= 1.0:
            continue
        # Cardoso et al. formula: sigma in mm
        variance = (k**2 - 1) * (2 * np.sqrt(2 * np.log(2))) ** (-2)
        sigma_mm = spacing[axis] * np.sqrt(variance)
        # Convert to voxels for the separable convolution
        sigmas[axis] = sigma_mm / spacing[axis]
    return sigmas


def _gaussian_smooth_batch(data: Tensor, sigmas: np.ndarray) -> Tensor:
    """Apply separable Gaussian smoothing along spatial axes of a 5D tensor.

    Args:
        data: `(B, C, I, J, K)` tensor.
        sigmas: Per-axis sigma in voxels. Zero means skip that axis.

    Returns:
        Smoothed `(B, C, I, J, K)` tensor.
    """
    result = data.float()
    b, c = result.shape[:2]
    for axis_idx in range(3):
        sigma = float(sigmas[axis_idx])
        if sigma <= 0:
            continue
        # Kernel radius: 3 sigma, at least 1
        radius = max(int(np.ceil(3 * sigma)), 1)
        kernel_size = 2 * radius + 1
        x = (
            torch.arange(
                kernel_size,
                dtype=torch.float32,
                device=data.device,
            )
            - radius
        )
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Build a 5D depthwise kernel for F.conv3d with groups=C.
        # Shape: (C, 1, kI, kJ, kK) with the kernel along the target axis.
        k_shape = [1, 1, 1]
        k_shape[axis_idx] = kernel_size
        kernel_3d = kernel_1d.reshape(1, 1, *k_shape).expand(c, 1, -1, -1, -1)

        # Replicate-pad along the target spatial axis.
        # F.pad order: (K_before, K_after, J_before, J_after, I_before, I_after)
        pad = [0] * 6
        pad_idx = 2 * (2 - axis_idx)
        pad[pad_idx] = radius
        pad[pad_idx + 1] = radius

        padded = functional.pad(result, pad, mode="replicate")
        # Convolve each (B, C, ...) slice with groups=C so channels are independent.
        result = functional.conv3d(
            rearrange(padded, "b c i j k -> (b c) 1 i j k"),
            kernel_3d[:1],
            padding=0,
        )
        result = rearrange(result, "(b c) 1 i j k -> b c i j k", b=b, c=c)
    return result.to(data.dtype)


def _batch_fill_value(
    img_batch: ImagesBatch,
    *,
    default_pad_value: TypePadValue | float,
    default_pad_label: float,
) -> float | Tensor:
    """Compute a single fill value for the whole image batch."""
    if issubclass(img_batch._image_class, LabelMap):
        return float(default_pad_label)

    if isinstance(default_pad_value, Number):
        return float(default_pad_value)

    if not isinstance(default_pad_value, str):
        msg = (
            "default_pad_value must be a string or number, got"
            f" {type(default_pad_value)}"
        )
        raise TypeError(msg)

    # Use the first sample to compute the channel-wise fill value
    first_tensor = img_batch.data[0]
    values = [
        _compute_channel_pad_value(channel, default_pad_value)
        for channel in first_tensor
    ]
    return torch.as_tensor(values, dtype=torch.float32)


def _prepare_fill_value(
    fill_value: float | Tensor,
    reference: Tensor,
) -> Tensor | None:
    """Convert a fill value to a broadcast-compatible tensor.

    Returns `None` when the fill is zero (so the caller can skip the
    masking step, since `grid_sample` already pads with zeros).
    """
    if isinstance(fill_value, Tensor):
        fill_tensor = fill_value.to(device=reference.device, dtype=reference.dtype)
    else:
        if float(fill_value) == 0.0:
            return None
        fill_tensor = torch.as_tensor(
            fill_value,
            device=reference.device,
            dtype=reference.dtype,
        )
    if fill_tensor.ndim == 0:
        return fill_tensor
    if fill_tensor.ndim == 1:
        return rearrange(fill_tensor, "c -> 1 c 1 1 1")
    return fill_tensor


def _compute_channel_pad_value(
    tensor: Tensor,
    default_pad_value: TypePadValue,
) -> float:
    """Compute a scalar fill value for a single 3D channel tensor."""
    if default_pad_value == "minimum":
        return float(tensor.min().item())
    if default_pad_value == "mean":
        return _border_mean(tensor, filter_otsu=False)
    if default_pad_value == "otsu":
        return _border_mean(tensor, filter_otsu=True)
    msg = f'Unknown default_pad_value "{default_pad_value}"'
    raise ValueError(msg)


def _border_mean(
    tensor: Tensor,
    *,
    filter_otsu: bool,
) -> float:
    """Mean intensity of the six boundary faces of a 3D tensor.

    When *filter_otsu* is `True`, only voxels below the Otsu threshold
    are averaged, giving a background-aware fill value.
    """
    borders = torch.cat(
        [
            tensor[0, :, :].ravel(),
            tensor[-1, :, :].ravel(),
            tensor[:, 0, :].ravel(),
            tensor[:, -1, :].ravel(),
            tensor[:, :, 0].ravel(),
            tensor[:, :, -1].ravel(),
        ]
    ).float()
    if not filter_otsu:
        return float(borders.mean().item())
    threshold = _otsu_threshold(borders)
    values = borders[borders < threshold]
    if values.numel() > 0:
        return float(values.mean().item())
    return float(borders.mean().item())


def _otsu_threshold(values: Tensor) -> float:
    """Compute the Otsu threshold for a 1D tensor of values.

    Sweeps over sorted values, maximizing the between-class variance
    to find the threshold that best separates foreground from background.
    """
    sorted_values, _ = values.sort()
    num_values = sorted_values.numel()
    if num_values == 0:
        return 0.0

    total_sum = float(sorted_values.sum().item())
    best_threshold = float(sorted_values[0].item())
    best_variance = 0.0
    background_sum = 0.0
    background_count = 0

    for background_count, item in enumerate(sorted_values[:-1], start=1):
        value = float(item.item())
        foreground_count = num_values - background_count
        background_sum += value

        mean_background = background_sum / background_count
        mean_foreground = (total_sum - background_sum) / foreground_count
        weight_background = background_count / num_values
        weight_foreground = foreground_count / num_values
        # Between-class variance: maximize this to find the best split.
        between_variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if between_variance > best_variance:
            best_variance = between_variance
            best_threshold = value
    return best_threshold


def _upsample_displacement_field(
    control_points: Tensor,
    output_shape: TypeThreeInts,
) -> Tensor:
    """Trilinearly upsample a coarse `(n_i, n_j, n_k, 3)` field.

    The result has shape `(*output_shape, 3)` and approximates
    cubic B-spline interpolation for smooth deformations.
    """
    # (I, J, K, 3) -> (1, 3, I, J, K) for interpolate
    field = rearrange(control_points, "i j k d -> 1 d i j k").float()
    dense = functional.interpolate(
        field,
        size=list(output_shape),
        mode="trilinear",
        align_corners=True,
    )
    # (1, 3, I, J, K) -> (I, J, K, 3)
    return rearrange(dense, "1 d i j k -> i j k d")


def _check_folding(
    control_points: np.ndarray,
    max_displacement: tuple[float, float, float],
    shape: TypeThreeInts,
    spacing: np.ndarray,
) -> None:
    """Warn if the displacement magnitude may cause grid folding.

    Folding occurs when a control point moves past its neighbor,
    inverting the local Jacobian.  The heuristic checks whether the
    maximum displacement exceeds half the coarse-grid spacing.
    """
    num_control_points = np.array(control_points.shape[:-1], dtype=np.float64)
    image_bounds = np.array(shape, dtype=np.float64) * spacing
    mesh_shape = num_control_points - _SPLINE_ORDER
    grid_spacing = image_bounds / mesh_shape
    conflicts = np.array(max_displacement, dtype=np.float64) > grid_spacing / 2
    if np.any(conflicts):
        (where,) = np.where(conflicts)
        warnings.warn(
            "The maximum displacement is larger than half the coarse-grid"
            f" spacing for dimensions {where.tolist()}, so folding may occur",
            RuntimeWarning,
            stacklevel=3,
        )


def _resolve_control_points(
    control_points: Tensor | None,
    num_control_points: TypeThreeInts,
    max_displacement: _ParameterRange,
    locked_borders: int,
) -> tuple[Tensor | None, tuple[float, float, float] | None]:
    """Return a concrete control-point field and its max displacement.

    If *control_points* is already provided, clone it.  Otherwise sample
    a random field from the displacement range.  Returns `(None, None)`
    when the sampled displacement is zero everywhere.
    """
    if control_points is not None:
        return control_points.clone(), _max_abs_displacement(control_points)

    sampled = max_displacement.sample()
    if all(value == 0.0 for value in sampled):
        return None, None
    field = _sample_control_points(num_control_points, sampled, locked_borders)
    return field, sampled


def _sample_control_points(
    grid_shape: TypeThreeInts,
    max_displacement: tuple[float, float, float],
    locked_borders: int,
) -> Tensor:
    """Sample a random control-point displacement field.

    Each component is drawn uniformly from `[-max, +max]` along each
    axis, then the outermost *locked_borders* layers are zeroed to
    prevent boundary artifacts.
    """
    field = torch.rand(*grid_shape, 3, dtype=torch.float32)
    field -= 0.5
    field *= 2
    for axis in range(3):
        field[..., axis] *= max_displacement[axis]

    # Zero out outermost control-point layers to avoid boundary artifacts.
    for border in range(locked_borders):
        field[border, :] = 0
        field[-1 - border, :] = 0
        field[:, border] = 0
        field[:, -1 - border] = 0
        field[:, :, border] = 0
        field[:, :, -1 - border] = 0
    return field


def _build_forward_affine(
    *,
    scales: tuple[float, float, float],
    degrees: tuple[float, float, float],
    translation: tuple[float, float, float],
    center: TypeCenter,
    shape: TypeThreeInts,
    affine: AffineMatrix,
) -> np.ndarray:
    """Build a 4x4 world-space affine from scale, rotation, translation.

    When *center* is `"image"`, the rotation and scaling pivot around
    the image center.  For 2D slices (last axis size 1), out-of-plane
    components are suppressed.
    """
    scaling = np.asarray(scales, dtype=np.float64)
    rotation = np.asarray(degrees, dtype=np.float64)
    shift = np.asarray(translation, dtype=np.float64)

    # Suppress out-of-plane components for 2D (single-slice) images.
    if shape[-1] == 1:
        scaling[2] = 1.0
        rotation[0] = 0.0
        rotation[1] = 0.0
        shift[2] = 0.0

    center_world = _image_center_world(shape, affine) if center == "image" else None
    return _physical_affine_matrix(
        scales=scaling,
        degrees=rotation,
        translation=shift,
        center_world=center_world,
    )


def _physical_affine_matrix(
    *,
    scales: np.ndarray,
    degrees: np.ndarray,
    translation: np.ndarray,
    center_world: np.ndarray | None,
) -> np.ndarray:
    """Compose rotation, scaling, and translation into a 4x4 matrix.

    If *center_world* is given the transform pivots around that point:
    `T = R @ S` with `t = center - R @ S @ center + translation`.
    """
    rotation = _euler_to_rotation_matrix(degrees)
    scale = np.diag(scales)
    transform = np.eye(4, dtype=np.float64)
    rotation_scale = rotation @ scale
    transform[:3, :3] = rotation_scale
    # Pivot: translate so center is at origin, apply R@S, translate back.
    if center_world is not None:
        transform[:3, 3] = center_world - rotation_scale @ center_world
    transform[:3, 3] += translation
    return transform


def _euler_to_rotation_matrix(degrees: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles in degrees to a 3x3 rotation matrix.

    Uses the ZYX extrinsic (= XYZ intrinsic) convention:
    `R = Rz @ Ry @ Rx`.
    """
    radians = np.radians(degrees)
    rx, ry, rz = radians

    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    rotation_x = np.array(
        [
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x],
        ],
        dtype=np.float64,
    )
    rotation_y = np.array(
        [
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y],
        ],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x


def _image_center_world(
    shape: TypeThreeInts,
    affine: AffineMatrix,
) -> np.ndarray:
    """Return the world-space coordinates of the image center."""
    center_index = (np.asarray(shape, dtype=np.float64) - 1) / 2
    matrix = affine.numpy()
    return matrix[:3, 3] + matrix[:3, :3] @ center_index


def _check_shared_space(
    images: dict[str, ImagesBatch],
    reference_shape: TypeThreeInts,
    reference_affine: AffineMatrix,
) -> None:
    """Raise if any image has a different shape or affine than the first."""
    reference_matrix = reference_affine.data
    for name, img_batch in images.items():
        current_shape = _get_spatial_shape(img_batch)
        if current_shape != reference_shape:
            msg = (
                f'Image "{name}" has shape {current_shape}, expected {reference_shape}'
            )
            raise RuntimeError(msg)
        for affine in img_batch.affines:
            if not torch.allclose(
                affine.data,
                reference_matrix,
                rtol=1e-6,
                atol=1e-6,
            ):
                msg = (
                    "Spatial transforms with affine or elastic components require"
                    " selected images to share the same affine"
                )
                raise RuntimeError(msg)


def _get_spatial_shape(img_batch: ImagesBatch) -> TypeThreeInts:
    """Extract the `(I, J, K)` spatial shape from a batched image."""
    return (
        int(img_batch.data.shape[-3]),
        int(img_batch.data.shape[-2]),
        int(img_batch.data.shape[-1]),
    )


def _interpolation_for_batch(
    img_batch: ImagesBatch,
    *,
    image_interpolation: TypeImageInterpolation,
    label_interpolation: TypeLabelInterpolation,
) -> str:
    """Choose the interpolation mode based on the image class."""
    if issubclass(img_batch._image_class, LabelMap):
        return label_interpolation
    return image_interpolation


def _serialize_space(space: TypeTargetSpace | None) -> dict[str, Any] | None:
    """Convert a `(shape, AffineMatrix)` pair to a JSON-safe dict."""
    if space is None:
        return None
    shape, affine = space
    return {
        "shape": list(shape),
        "affine": affine.numpy().tolist(),
    }


def _deserialize_space(data: dict[str, Any] | None) -> TypeTargetSpace | None:
    """Reconstruct a `(shape, AffineMatrix)` pair from a dict."""
    if data is None:
        return None
    shape = (
        int(data["shape"][0]),
        int(data["shape"][1]),
        int(data["shape"][2]),
    )
    return shape, AffineMatrix(np.asarray(data["affine"], dtype=np.float64))


def _serialize_matrix(matrix: np.ndarray | None) -> list[list[float]] | None:
    """Convert a numpy matrix to a nested list for JSON serialization."""
    if matrix is None:
        return None
    return matrix.tolist()


def _deserialize_matrix(data: list[list[float]] | None) -> np.ndarray | None:
    """Reconstruct a numpy matrix from a nested list."""
    if data is None:
        return None
    return np.asarray(data, dtype=np.float64)


def _serialize_control_points(control_points: Tensor | None) -> list | None:
    """Convert a control-point tensor to a nested list."""
    if control_points is None:
        return None
    return control_points.cpu().tolist()


def _deserialize_control_points(data: list | None) -> Tensor | None:
    """Reconstruct a control-point tensor from a nested list."""
    if data is None:
        return None
    return torch.as_tensor(data, dtype=torch.float32)


def _deserialize_max_displacement(
    values: list[float] | None,
) -> tuple[float, float, float] | None:
    """Reconstruct a max-displacement 3-tuple from a list."""
    if values is None:
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def _max_abs_displacement(control_points: Tensor) -> tuple[float, float, float]:
    """Return the per-axis maximum absolute displacement."""
    absolute = control_points.abs()
    return (
        float(absolute[..., 0].max().item()),
        float(absolute[..., 1].max().item()),
        float(absolute[..., 2].max().item()),
    )


def _sample_scales(
    scales: _ParameterRange,
    isotropic: bool,
) -> tuple[float, float, float]:
    """Sample a 3-tuple of scale factors, optionally isotropic."""
    if isotropic:
        value = scales.sample_1d()
        return (value, value, value)
    return scales.sample()


def _has_affine_component(
    scales: tuple[float, float, float],
    degrees: tuple[float, float, float],
    translation: tuple[float, float, float],
) -> bool:
    """Return `True` if any sampled affine parameter is non-identity."""
    return not (
        np.allclose(scales, (1.0, 1.0, 1.0))
        and np.allclose(degrees, (0.0, 0.0, 0.0))
        and np.allclose(translation, (0.0, 0.0, 0.0))
    )


def _parse_target_space_tuple(
    shape: Sequence[int],
    affine: AffineMatrix | Tensor | npt.ArrayLike,
) -> TypeTargetSpace:
    """Validate and convert a `(shape, affine)` target pair."""
    if len(shape) != 3:
        msg = f"Target shape must have length 3, got {len(shape)}"
        raise ValueError(msg)
    target_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
    return target_shape, AffineMatrix(affine)


def _is_spacing_sequence(target: Sequence[Any]) -> bool:
    """Return `True` if *target* looks like a 3-element numeric spacing."""
    return len(target) == 3 and all(isinstance(value, Number) for value in target)


def _is_spacing_tuple(
    target: object,
) -> TypeGuard[tuple[int | float, int | float, int | float]]:
    """Type-guard: *target* is a 3-number tuple."""
    return isinstance(target, tuple) and _is_spacing_sequence(target)


def _is_spacing_list(target: object) -> TypeGuard[list[int | float]]:
    """Type-guard: *target* is a 3-number list."""
    return isinstance(target, list) and _is_spacing_sequence(target)


def _is_target_space_tuple(
    target: object,
) -> TypeGuard[tuple[Sequence[int], AffineMatrix | Tensor | npt.ArrayLike]]:
    """Type-guard: *target* is a 2-element `(shape, affine)` tuple.

    The first element of a target-space tuple is a shape sequence, never a
    plain number, so a 2-tuple of numbers like `(2, 4)` is treated as a
    spacing range rather than a `(shape, affine)` pair.
    """
    return (
        isinstance(target, tuple)
        and len(target) == 2
        and not isinstance(target[0], Number)
    )


def _parse_spacing(
    value: TypeSpacing | Sequence[float] | np.ndarray | float | int,
) -> TypeSpacing:
    """Normalize a spacing specification to a strictly-positive 3-tuple."""
    if isinstance(value, (int, float)):
        spacing = (float(value), float(value), float(value))
    elif isinstance(value, np.ndarray):
        if value.size != 3:
            msg = f"Spacing array must have 3 values, got {value.size}"
            raise ValueError(msg)
        flat = [float(v) for v in value.flat]
        spacing = (flat[0], flat[1], flat[2])
    else:
        if len(value) != 3:
            msg = f"Spacing must have 3 values, got {len(value)}"
            raise ValueError(msg)
        spacing = (float(value[0]), float(value[1]), float(value[2]))
    if any(v <= 0 for v in spacing):
        msg = f"Spacing must be strictly positive, got {spacing}"
        raise ValueError(msg)
    return spacing


def _parse_interpolation(
    interpolation: TypeInterpolation | int,
) -> TypeInterpolation:
    """Validate an interpolation mode (string or integer order).

    Integer orders (0-7) are converted to their string equivalents.
    """
    if isinstance(interpolation, int):
        order_to_name = {v: k for k, v in _INTERPOLATION_TO_ORDER.items()}
        if interpolation not in order_to_name:
            msg = f"Interpolation order {interpolation} is not supported. Must be 0-7."
            raise ValueError(msg)
        return cast(TypeInterpolation, order_to_name[interpolation])
    if not isinstance(interpolation, str):
        msg = f"Interpolation must be a string or int, got {type(interpolation)}"
        raise TypeError(msg)
    lowered = interpolation.lower()
    if lowered not in _SUPPORTED_INTERPOLATIONS:
        msg = (
            f'Interpolation "{lowered}" is not supported. Supported values are'
            f" {_SUPPORTED_INTERPOLATIONS}"
        )
        raise ValueError(msg)
    return lowered


def _parse_one_hot_label_interpolation(
    interpolation: TypeImageInterpolation | int,
) -> TypeImageInterpolation:
    """Validate the per-channel interpolation used by the `"label"` mode.

    Accepts the same modes as image interpolation (`"nearest"`, `"linear"`,
    or B-spline orders `"quadratic"`-`"seventh"` / integer orders 0-7) but
    rejects `"label"`, which would recurse.

    Args:
        interpolation: Interpolation mode (string or integer order).

    Returns:
        The validated interpolation mode as a lowercase string.

    Raises:
        ValueError: If `"label"` is passed.
    """
    parsed = _parse_interpolation(interpolation)
    if parsed == LABEL_INTERPOLATION:
        msg = (
            f'one_hot_label_interpolation cannot be "{LABEL_INTERPOLATION}"; choose'
            ' an interpolation for the one-hot channels (e.g. "linear")'
        )
        raise ValueError(msg)
    return parsed


def _parse_default_pad_value(value: TypePadValue | float) -> TypePadValue | float:
    """Validate a pad-value specification (string keyword or number)."""
    if isinstance(value, Number):
        return float(value)
    if value in _SUPPORTED_PAD_VALUES:
        return value
    msg = 'default_pad_value must be "minimum", "mean", "otsu", or a numeric value'
    raise ValueError(msg)


def _parse_center(center: TypeCenter) -> TypeCenter:
    """Validate the *center* argument."""
    if center not in ("image", "origin"):
        msg = f'center must be "image" or "origin", got "{center}"'
        raise ValueError(msg)
    return center


def _to_positive_range(
    value: TypeParameterValue,
) -> _ParameterRange:
    """Convert to a `_ParameterRange`, rejecting non-positive scales."""
    result = _to_parameter_range(value)
    if result._distribution is None:
        for low, high in result._ranges:
            if low <= 0 or high <= 0:
                msg = f"Scale factors must be strictly positive, got {value}"
                raise ValueError(msg)
    return result


def _validate_isotropic(
    value: TypeParameterValue,
    isotropic: bool,
) -> None:
    """Raise if *isotropic* is `True` but per-axis values were given."""
    if not isotropic or isinstance(value, Distribution):
        return
    if isinstance(value, tuple) and len(value) in (3, 6):
        msg = "If isotropic=True, scales must be a single value or a 2-value range"
        raise ValueError(msg)


def _parse_num_control_points(
    value: int | TypeThreeInts,
) -> TypeThreeInts:
    """Normalize to a 3-tuple and validate each axis has >= 4 points."""
    parsed = (value, value, value) if isinstance(value, int) else value
    for axis, number in enumerate(parsed):
        if not isinstance(number, int) or number < 4:
            msg = (
                "Each num_control_points value must be an integer greater than 3;"
                f" axis {axis} got {number}"
            )
            raise ValueError(msg)
    return parsed


def _parse_locked_borders(value: int) -> int:
    """Validate that *value* is 0, 1, or 2."""
    if value not in (0, 1, 2):
        msg = f"locked_borders must be 0, 1, or 2, got {value}"
        raise ValueError(msg)
    return value


def _parse_control_points(control_points: TypeControlPoints) -> Tensor:
    """Validate and convert a control-point field to a contiguous float32 tensor."""
    tensor = (
        control_points.clone().detach().to(torch.float32)
        if isinstance(control_points, Tensor)
        else torch.as_tensor(np.asarray(control_points), dtype=torch.float32)
    )
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        msg = (
            "control_points must have shape (n_i, n_j, n_k, 3), got"
            f" {tuple(tensor.shape)}"
        )
        raise ValueError(msg)
    for axis, size in enumerate(tensor.shape[:-1]):
        if size < 4:
            msg = (
                "Each control-point axis must have at least 4 elements;"
                f" axis {axis} got {size}"
            )
            raise ValueError(msg)
    return tensor.contiguous()


def _normalize_parameter_value(
    value: TypeParameterValue,
) -> float | tuple | Distribution | Choice:
    """Cast ints to floats so `_ParameterRange` always receives floats."""
    if isinstance(value, (Distribution, Choice)):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    # Tuple: may contain mixed specs (Choice, Distribution, sub-tuples)
    # so pass through if any element is non-numeric.
    if isinstance(value, tuple):
        if all(isinstance(v, (int, float)) for v in value):
            return tuple(float(v) for v in value)
        return value
    return value


def _to_parameter_range(value: TypeParameterValue) -> _ParameterRange:
    """Convert a `TypeParameterValue` to a `_ParameterRange`."""
    return _ParameterRange(_normalize_parameter_value(value))


def _to_nonnegative_parameter_range(value: TypeParameterValue) -> _ParameterRange:
    """Like `_to_parameter_range`, but rejects negative values."""
    result = _to_parameter_range(value)
    if result._distribution is None:
        for low, high in result._ranges:
            if low < 0 or high < 0:
                msg = f"Value must be non-negative, got {value}"
                raise ValueError(msg)
    return result
