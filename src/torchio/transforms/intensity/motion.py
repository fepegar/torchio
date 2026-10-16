"""Motion: simulate MRI motion artifacts via k-space corruption."""

from __future__ import annotations

from functools import partial
from operator import itemgetter
from typing import Any

import torch
import torch.nn.functional as functional
from einops import rearrange
from einops import repeat
from torch import Tensor

from ...data.batch import SubjectsBatch
from ..parameter_range import to_range
from ..transform import IntensityTransform

RigidTransform = dict[str, tuple[float, float, float]]
MotionTransforms = list[RigidTransform]
PerElementMotionTransforms = list[MotionTransforms]
MotionParameters = tuple[Tensor, Tensor]

_IDENTITY_TRANSFORM: RigidTransform = {
    "degrees": (0.0, 0.0, 0.0),
    "translation": (0.0, 0.0, 0.0),
}


class Motion(IntensityTransform):
    r"""Simulate MRI motion artifacts.

    Motion during MR acquisition corrupts different segments of
    k-space with different rigid-body transforms, producing
    characteristic ringing and blurring.  This implementation follows
    [Shaw et al., 2019](http://proceedings.mlr.press/v102/shaw19a.html).

    The simulation:

    1. Splits k-space into *num_transforms* + 1 segments along a
       random axis.
    2. For each segment, applies a random rigid-body transform to
       the image and fills the corresponding k-space lines from the
       transformed image.
    3. Reconstructs the corrupted image via inverse FFT.

    Args:
        degrees: Rotation range in degrees.  A scalar $d$ means
            $\theta_i \sim \mathcal{U}(-d, d)$.  A 2-tuple $(a, b)$
            means $\theta_i \sim \mathcal{U}(a, b)$.
        translation: Translation range in voxels, same convention as
            *degrees*. The translation is applied in normalized grid
            coordinates (a voxel-space approximation), not in millimeters.
        num_transforms: Number of inter-segment motion events.
            More transforms produce more distortion.
        axes: Spatial axes along which k-space may be split into
            segments.  One is chosen at random per application.  Pass
            a single-element tuple such as ``axes=(0,)`` to always
            split along the same axis.
        **kwargs: See [`Transform`][torchio.Transform].

    Warning:
        Large numbers of transforms increase execution time
        significantly for 3D volumes.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Motion()
        >>> transform = tio.Motion(degrees=15, translation=10, num_transforms=4)
    """

    def __init__(
        self,
        *,
        degrees: float | tuple[float, float] = 10.0,
        translation: float | tuple[float, float] = 10.0,
        num_transforms: int = 2,
        axes: tuple[int, ...] = (0, 1, 2),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.degrees = to_range(degrees)
        self.translation = to_range(translation)
        if not isinstance(num_transforms, int) or num_transforms < 1:
            msg = f"num_transforms must be a positive int, got {num_transforms}"
            raise ValueError(msg)
        self.num_transforms = num_transforms
        if not axes or any(axis not in (0, 1, 2) for axis in axes):
            msg = f"axes must be a non-empty tuple of values in (0, 1, 2), got {axes}"
            raise ValueError(msg)
        self.axes = axes

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample motion parameters (per element when batched)."""
        n = self._resolve_n(batch)
        if n is None:
            transforms = self._sample_transforms()
            return {"transforms": transforms, "axis": self._sample_axis()}
        keep = self._keep_mask(batch, n)
        transforms_list: list[Any] = []
        axis_list: list[int] = []
        for index in range(n):
            if keep is not None and not keep[index]:
                transforms_list.append([])
                axis_list.append(self.axes[0])
                continue
            transforms_list.append(self._sample_transforms())
            axis_list.append(self._sample_axis())
        params = {"transforms": transforms_list, "axis": axis_list}
        self._tag_batched(params, batch, n, keep, ["transforms", "axis"])
        return params

    def _sample_axis(self) -> int:
        """Sample one k-space segmentation axis."""
        return self.axes[int(torch.randint(len(self.axes), (1,)).item())]

    def _sample_transforms(self) -> MotionTransforms:
        """Sample one list of rigid sub-transforms."""
        return [
            {
                "degrees": self.degrees.sample(),
                "translation": self.translation.sample(),
            }
            for _ in range(self.num_transforms)
        ]

    @property
    def supports_per_instance_params(self) -> bool:
        return True

    @property
    def supports_per_instance_p(self) -> bool:
        return True

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Corrupt each selected image with simulated motion."""
        per_instance = self._is_per_instance_params(params)
        for _name, img_batch in self._get_images(batch).items():
            if per_instance:
                img_batch.data = _apply_motion_per_instance(
                    img_batch.data,
                    params["transforms"],
                    params["axis"],
                )
            else:
                img_batch.data = _apply_motion(
                    img_batch.data,
                    params["transforms"],
                    params["axis"],
                )
        return batch


def _apply_motion(
    data: Tensor,
    motion_transforms: MotionTransforms,
    axis: int,
) -> Tensor:
    """Apply motion corruption to a 5D tensor.

    Args:
        data: `(B, C, I, J, K)` image tensor.
        motion_transforms: List of dicts with `degrees` and
            `translation` 3-tuples.
        axis: Shared spatial axis along which k-space is segmented.

    Returns:
        Motion-corrupted `(B, C, I, J, K)` tensor.
    """
    if not motion_transforms:
        return data
    batch_size = data.shape[0]
    segment_parameters = [
        _shared_motion_parameters(transform, batch_size, data.device)
        for transform in motion_transforms
    ]
    axes = torch.full((batch_size,), axis, dtype=torch.long, device=data.device)
    return _apply_motion_segments(data, segment_parameters, axes)


def _apply_motion_per_instance(
    data: Tensor,
    motion_transforms: PerElementMotionTransforms,
    axis: list[int],
) -> Tensor:
    """Apply motion corruption with per-element rigid parameters.

    Args:
        data: `(B, C, I, J, K)` image tensor.
        motion_transforms: One transform list per batch element. Empty
            lists mark gated-out elements.
        axis: Per-element spatial axis along which k-space is segmented.

    Returns:
        Motion-corrupted `(B, C, I, J, K)` tensor, with inactive rows
            restored exactly from the input.
    """
    _check_batch_size(data, motion_transforms)
    active = _active_motion_mask(motion_transforms, data.device)
    if not active.any().item():
        return data
    segment_parameters = [
        _per_instance_motion_parameters(motion_transforms, segment_index, data.device)
        for segment_index in range(_num_motion_transforms(motion_transforms))
    ]
    axes = torch.as_tensor(axis, dtype=torch.long, device=data.device)
    # Inactive rows must not introduce additional segment-size constraints.
    axes = torch.where(active, axes, axes[active][0])
    transformed = _apply_motion_segments(data, segment_parameters, axes)
    active = rearrange(active, "b -> b 1 1 1 1")
    return torch.where(active, transformed, data)


def _check_batch_size(
    data: Tensor,
    motion_transforms: PerElementMotionTransforms,
) -> None:
    """Validate that parameter lists match the batch size.

    Args:
        data: `(B, C, I, J, K)` image tensor.
        motion_transforms: One transform list per batch element.

    Raises:
        ValueError: If the parameter count differs from the batch size.
    """
    if len(motion_transforms) == data.shape[0]:
        return
    msg = (
        f"Expected {data.shape[0]} motion parameter lists, got {len(motion_transforms)}"
    )
    raise ValueError(msg)


def _active_motion_mask(
    motion_transforms: PerElementMotionTransforms,
    device: torch.device,
) -> Tensor:
    """Return a boolean mask for elements with sampled motion.

    Args:
        motion_transforms: One transform list per batch element.
        device: Device where the mask will be allocated.

    Returns:
        Boolean `(B,)` tensor.
    """
    return torch.as_tensor(
        tuple(map(bool, motion_transforms)),
        dtype=torch.bool,
        device=device,
    )


def _num_motion_transforms(
    motion_transforms: PerElementMotionTransforms,
) -> int:
    """Return the uniform number of transforms for active elements.

    Args:
        motion_transforms: One transform list per batch element.

    Returns:
        Number of rigid transforms per active element.

    Raises:
        ValueError: If active elements have inconsistent transform counts.
    """
    lengths = set(map(len, motion_transforms))
    lengths.discard(0)
    if len(lengths) <= 1:
        return max(lengths, default=0)
    msg = f"Expected uniform motion transform counts, got {sorted(lengths)}"
    raise ValueError(msg)


def _shared_motion_parameters(
    motion_transform: RigidTransform,
    batch_size: int,
    device: torch.device,
) -> MotionParameters:
    """Convert a shared rigid transform into batched tensors.

    Args:
        motion_transform: Shared rigid transform parameters.
        batch_size: Number of batch elements.
        device: Device where tensors will be allocated.

    Returns:
        Batched degrees and translation tensors of shape `(B, 3)`.
    """
    degrees = _repeat_parameter(motion_transform["degrees"], batch_size, device)
    translation = _repeat_parameter(
        motion_transform["translation"],
        batch_size,
        device,
    )
    return degrees, translation


def _per_instance_motion_parameters(
    motion_transforms: PerElementMotionTransforms,
    segment_index: int,
    device: torch.device,
) -> MotionParameters:
    """Collect one segment's per-element parameters as tensors.

    Args:
        motion_transforms: One transform list per batch element.
        segment_index: Segment transform index.
        device: Device where tensors will be allocated.

    Returns:
        Batched degrees and translation tensors of shape `(B, 3)`.
    """
    get_segment_transform = partial(
        _segment_transform_or_identity,
        segment_index=segment_index,
    )
    segment_transforms = tuple(map(get_segment_transform, motion_transforms))
    degrees = torch.as_tensor(
        tuple(map(itemgetter("degrees"), segment_transforms)),
        dtype=torch.float32,
        device=device,
    )
    translation = torch.as_tensor(
        tuple(map(itemgetter("translation"), segment_transforms)),
        dtype=torch.float32,
        device=device,
    )
    return degrees, translation


def _segment_transform_or_identity(
    transforms: MotionTransforms,
    *,
    segment_index: int,
) -> RigidTransform:
    """Return segment parameters or identity for inactive elements.

    Args:
        transforms: Rigid transform list for one batch element.
        segment_index: Segment transform index.

    Returns:
        The segment's rigid transform or identity parameters.
    """
    if not transforms:
        return _IDENTITY_TRANSFORM
    return transforms[segment_index]


def _repeat_parameter(
    parameter: tuple[float, float, float],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Repeat a shared 3-vector parameter across the batch.

    Args:
        parameter: Shared 3-vector parameter.
        batch_size: Number of batch elements.
        device: Device where the result will be allocated.

    Returns:
        `(B, 3)` float tensor.
    """
    tensor = torch.as_tensor(parameter, dtype=torch.float32, device=device)
    return repeat(tensor, "component -> batch component", batch=batch_size)


def _apply_motion_segments(
    data: Tensor,
    segment_parameters: list[MotionParameters],
    axes: Tensor,
) -> Tensor:
    """Apply k-space segment replacements for a whole batch.

    Args:
        data: `(B, C, I, J, K)` image tensor.
        segment_parameters: Per-segment batched degrees and translations.
        axes: `(B,)` spatial axis per batch element along which k-space
            is split into segments.

    Returns:
        Motion-corrupted `(B, C, I, J, K)` tensor.
    """
    result = data.float()
    num_segments = len(segment_parameters) + 1
    spatial_shape = result.shape[-3:]
    _check_segment_sizes(spatial_shape, axes, num_segments)
    spectrum = torch.fft.fftn(result, dim=(-3, -2, -1))
    for segment_index, (degrees, translation) in enumerate(
        segment_parameters,
        start=1,
    ):
        moved = _apply_rigid_transform(result, degrees, translation)
        moved_spectrum = torch.fft.fftn(moved, dim=(-3, -2, -1))
        mask = _segment_mask(
            segment_index,
            num_segments,
            spatial_shape,
            axes,
            data.device,
        )
        spectrum = torch.where(mask, moved_spectrum, spectrum)

    reconstructed = torch.fft.ifftn(spectrum, dim=(-3, -2, -1)).real
    return reconstructed.to(data.dtype)


def _check_segment_sizes(
    spatial_shape: torch.Size,
    axes: Tensor,
    num_segments: int,
) -> None:
    """Validate that every used axis can hold the requested segments.

    Args:
        spatial_shape: `(I, J, K)` spatial sizes.
        axes: `(B,)` spatial axis per batch element.
        num_segments: Total number of k-space segments.

    Raises:
        ValueError: If any used axis is too small to be split into
            `num_segments` non-empty segments.
    """
    for axis in range(3):
        if not bool((axes == axis).any()):
            continue
        if spatial_shape[axis] < num_segments:
            msg = (
                f"Cannot split {spatial_shape[axis]} k-space slices along"
                f" spatial axis {axis} into {num_segments} motion segments;"
                " reduce num_transforms or use a larger image along that axis."
            )
            raise ValueError(msg)


def _segment_mask(
    segment_index: int,
    num_segments: int,
    spatial_shape: torch.Size,
    axes: Tensor,
    device: torch.device,
) -> Tensor:
    """Build a `(B, 1, I, J, K)` mask for one k-space segment.

    For each batch element the segment spans `[start, end)` along that
    element's chosen axis and the full extent of the other axes.

    Args:
        segment_index: One-based segment index.
        num_segments: Total number of k-space segments.
        spatial_shape: `(I, J, K)` spatial sizes.
        axes: `(B,)` spatial axis per batch element.
        device: Device where the mask will be allocated.

    Returns:
        Boolean `(B, 1, I, J, K)` mask.
    """
    batch_size = axes.shape[0]
    mask = torch.zeros((batch_size, 1, *spatial_shape), dtype=torch.bool, device=device)
    for axis in range(3):
        selected = axes == axis
        if not bool(selected.any()):
            continue
        size = spatial_shape[axis]
        segment_size = size // num_segments
        start = segment_index * segment_size
        end = size if segment_index == num_segments - 1 else start + segment_size
        line = torch.zeros(size, dtype=torch.bool, device=device)
        line[start:end] = True
        line_shape = [1, 1, 1, 1, 1]
        line_shape[axis + 2] = size
        line = line.view(line_shape)
        mask |= selected.view(batch_size, 1, 1, 1, 1) & line
    return mask


def _apply_rigid_transform(
    tensor: Tensor,
    degrees: Tensor,
    translation: Tensor,
) -> Tensor:
    """Apply per-element rigid-body transforms to a 5-D tensor.

    Each batch element gets its own affine grid, shared by all channels.

    Args:
        tensor: `(B, C, I, J, K)` tensor.
        degrees: Euler angles in degrees, with shape `(B, 3)`.
        translation: Translation in voxels, with shape `(B, 3)`.

    Returns:
        Transformed `(B, C, I, J, K)` tensor.
    """
    batch_size, channels, *shape = tensor.shape
    theta = _affine_matrices(degrees, translation, shape)
    grid = functional.affine_grid(
        theta,
        [batch_size, 1, shape[0], shape[1], shape[2]],
        align_corners=True,
    )
    input_5d = rearrange(tensor, "b c i j k -> (b c) 1 i j k").float()
    grid = repeat(grid, "b i j k xyz -> (b c) i j k xyz", c=channels)
    output = functional.grid_sample(
        input_5d,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return rearrange(output, "(b c) 1 i j k -> b c i j k", b=batch_size)


def _affine_matrices(
    degrees: Tensor,
    translation: Tensor,
    spatial_shape: list[int],
) -> Tensor:
    """Build batched affine matrices for `affine_grid`.

    Args:
        degrees: Euler angles in degrees, with shape `(B, 3)`.
        translation: Translation in voxels, with shape `(B, 3)`.
        spatial_shape: Spatial tensor shape `(I, J, K)`.

    Returns:
        Batched affine matrices with shape `(B, 3, 4)`.
    """
    theta = torch.zeros(
        degrees.shape[0],
        3,
        4,
        dtype=degrees.dtype,
        device=degrees.device,
    )
    theta[:, :3, :3] = _rotation_matrices(degrees)
    theta[:, :3, 3] = _normalized_translation(translation, spatial_shape)
    return theta


def _normalized_translation(
    translation: Tensor,
    spatial_shape: list[int],
) -> Tensor:
    """Normalize voxel translations to `affine_grid` coordinates.

    Args:
        translation: Translation in voxels, with shape `(B, 3)`.
        spatial_shape: Spatial tensor shape `(I, J, K)`.

    Returns:
        Normalized translations with shape `(B, 3)`.
    """
    shape = torch.as_tensor(
        spatial_shape,
        dtype=translation.dtype,
        device=translation.device,
    )
    return translation / (shape / 2)


def _rotation_matrices(degrees: Tensor) -> Tensor:
    """Build batched Euler rotation matrices.

    Args:
        degrees: Euler angles in degrees, with shape `(B, 3)`.

    Returns:
        Rotation matrices with shape `(B, 3, 3)`.
    """
    radians = torch.deg2rad(degrees)
    rx, ry, rz = radians.unbind(dim=-1)
    r_x = _axis_rotation_matrices(rx, axis=0)
    r_y = _axis_rotation_matrices(ry, axis=1)
    r_z = _axis_rotation_matrices(rz, axis=2)
    return r_z @ r_y @ r_x


def _axis_rotation_matrices(angles: Tensor, *, axis: int) -> Tensor:
    """Build batched rotation matrices around one axis.

    Args:
        angles: Rotation angles in radians, with shape `(B,)`.
        axis: Rotation axis, where 0, 1 and 2 are x, y and z.

    Returns:
        Rotation matrices with shape `(B, 3, 3)`.

    Raises:
        ValueError: If `axis` is not 0, 1 or 2.
    """
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    matrices = torch.zeros(
        angles.shape[0],
        3,
        3,
        dtype=angles.dtype,
        device=angles.device,
    )
    if axis == 0:
        matrices[:, 0, 0] = 1
        matrices[:, 1, 1] = cos
        matrices[:, 1, 2] = -sin
        matrices[:, 2, 1] = sin
        matrices[:, 2, 2] = cos
        return matrices
    if axis == 1:
        matrices[:, 0, 0] = cos
        matrices[:, 0, 2] = sin
        matrices[:, 1, 1] = 1
        matrices[:, 2, 0] = -sin
        matrices[:, 2, 2] = cos
        return matrices
    if axis == 2:
        matrices[:, 0, 0] = cos
        matrices[:, 0, 1] = -sin
        matrices[:, 1, 0] = sin
        matrices[:, 1, 1] = cos
        matrices[:, 2, 2] = 1
        return matrices
    msg = f"Expected axis to be 0, 1 or 2, got {axis}"
    raise ValueError(msg)
