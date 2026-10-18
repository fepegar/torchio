"""Anisotropy: simulate low-resolution acquisition along one axis."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional
from einops import rearrange

from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from ..parameter_range import to_nonneg_range
from ..transform import Transform


class Anisotropy(Transform):
    r"""Simulate an anisotropic acquisition.

    Downsample along a randomly chosen axis and then upsample back
    to the original shape, emulating the through-plane blur seen in
    clinical MRI when one axis has coarser resolution.

    This is useful as a data augmentation for super-resolution
    training.

    Args:
        axes: Spatial axes eligible for downsampling.  One is chosen
            at random per application.
        downsampling: Downsampling factor $m \geq 1$.  A scalar is
            deterministic; a 2-tuple $(a, b)$ samples
            $m \sim \mathcal{U}(a, b)$.  The default `downsampling=1`
            is a no-op (and warns).
        image_interpolation: Interpolation mode used when upsampling
            scalar images back to the original shape.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Anisotropy(downsampling=4)
        >>> transform = tio.Anisotropy(
        ...     axes=(2,),
        ...     downsampling=(1.5, 5),
        ... )
    """

    def __init__(
        self,
        *,
        axes: tuple[int, ...] = (0, 1, 2),
        downsampling: float | tuple[float, float] = 1.0,
        image_interpolation: str = "linear",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.axes = axes
        self.downsampling = to_nonneg_range(downsampling)
        self.image_interpolation = image_interpolation
        self._validate_downsampling()
        self._warn_if_noop(
            is_noop=self.downsampling.is_constant(1.0),
            hint="downsampling=(1.5, 5)",
        )

    def _validate_downsampling(self) -> None:
        """Ensure the range produces factors >= 1."""
        _lo, hi = self.downsampling._ranges[0]
        if hi < 1.0:
            msg = f"downsampling range upper bound must be >= 1, got {hi}"
            raise ValueError(msg)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample axis and downsampling factor (per element when batched)."""
        n = self._resolve_n(batch)
        if n is None:
            axis = self.axes[int(torch.randint(len(self.axes), (1,)).item())]
            factor = max(1.0, self.downsampling.sample_1d())
            return {"axis": axis, "factor": factor}
        keep = self._keep_mask(batch, n)
        axis_list: list[int] = []
        factor_list: list[float] = []
        for index in range(n):
            if keep is not None and not keep[index]:
                axis_list.append(self.axes[0])
                factor_list.append(1.0)
                continue
            axis_list.append(self.axes[int(torch.randint(len(self.axes), (1,)).item())])
            factor_list.append(max(1.0, self.downsampling.sample_1d()))
        params = {"axis": axis_list, "factor": factor_list}
        self._tag_batched(params, batch, n, keep, ["axis", "factor"])
        return params

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
        """Downsample then upsample along the chosen axis."""
        per_instance = self._is_per_instance_params(params)
        for _name, img_batch in self._get_images(batch).items():
            is_label = issubclass(img_batch._image_class, LabelMap)
            mode = "nearest" if is_label else self.image_interpolation
            if per_instance:
                data = img_batch.data
                img_batch.data = _simulate_anisotropy_per_instance(
                    data,
                    axes=params["axis"],
                    factors=params["factor"],
                    mode=mode,
                )
            else:
                factor = params["factor"]
                if factor <= 1.0:
                    continue
                img_batch.data = _simulate_anisotropy(
                    img_batch.data,
                    axis=params["axis"],
                    factor=factor,
                    mode=mode,
                )
        return batch


def _simulate_anisotropy_per_instance(
    data: torch.Tensor,
    *,
    axes: list[int],
    factors: list[float],
    mode: str,
) -> torch.Tensor:
    """Downsample then upsample each batch element with its own parameters.

    Args:
        data: `(B, C, I, J, K)` tensor.
        axes: Spatial axis per batch element.
        factors: Downsampling factor per batch element.
        mode: Interpolation mode for upsampling (`"nearest"` or
            `"linear"`).

    Returns:
        Degraded `(B, C, I, J, K)` tensor with original shape.
    """
    axes_tensor = torch.as_tensor(axes, dtype=torch.long, device=data.device)
    factors_tensor = torch.as_tensor(
        factors,
        dtype=torch.float64,
        device=data.device,
    )
    active = factors_tensor > 1.0
    if not bool(active.any()):
        return data.to(data.dtype)

    active_axes = axes_tensor[active]
    if bool(((active_axes < 0) | (active_axes > 2)).any()):
        msg = f"Anisotropy axis must be in {{0, 1, 2}}, got {sorted(set(axes))}"
        raise ValueError(msg)

    output = data.clone()
    for axis in range(3):
        axis_mask = active & (axes_tensor == axis)
        if not bool(axis_mask.any()):
            continue
        output[axis_mask] = _simulate_anisotropy_fixed_axis(
            data[axis_mask],
            factors=factors_tensor[axis_mask],
            axis=axis,
            mode=mode,
        )
    return output.to(data.dtype)


def _simulate_anisotropy_fixed_axis(
    data: torch.Tensor,
    *,
    factors: torch.Tensor,
    axis: int,
    mode: str,
) -> torch.Tensor:
    """Downsample then upsample one axis with per-element factors.

    Args:
        data: `(B, C, I, J, K)` tensor whose elements share `axis`.
        factors: Downsampling factors for the batch elements.
        axis: Spatial axis (0, 1, or 2) to degrade.
        mode: Interpolation mode for upsampling (`"nearest"` or
            `"linear"`).

    Returns:
        Degraded `(B, C, I, J, K)` tensor with original shape.
    """
    length = data.shape[axis + 2]
    down_sizes = _downsample_sizes(length, factors)
    if mode == "nearest":
        indices = _nearest_source_indices(length, down_sizes, data.device)
        return _gather_axis(data.float(), indices, axis).to(data.dtype)
    lower, upper, weights = _linear_source_indices(length, down_sizes, data.device)
    lower_values = _gather_axis(data.float(), lower, axis)
    upper_values = _gather_axis(data.float(), upper, axis)
    weights = _broadcast_axis_weights(weights, axis)
    degraded = lower_values * (1.0 - weights) + upper_values * weights
    return degraded.to(data.dtype)


def _downsample_sizes(length: int, factors: torch.Tensor) -> torch.Tensor:
    """Return PyTorch-compatible nearest-downsampled sizes.

    Args:
        length: Original length along the degraded axis.
        factors: Per-element downsampling factors.

    Returns:
        Downsampled sizes matching `round(length / factor)`.
    """
    sizes = torch.round(length / factors).clamp_min(1)
    return sizes.to(torch.long)


def _nearest_source_indices(
    length: int,
    down_sizes: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return original-axis indices for nearest downsample and upsample.

    Args:
        length: Original length along the degraded axis.
        down_sizes: Downsampled sizes per batch element.
        device: Device where the indices should be created.

    Returns:
        A `(B, length)` tensor of source indices.
    """
    positions = torch.arange(length, dtype=torch.long, device=device)
    lowres_indices = torch.div(
        positions * rearrange(down_sizes, "b -> b 1"),
        length,
        rounding_mode="floor",
    )
    return _downsample_source_indices(length, down_sizes, lowres_indices)


def _linear_source_indices(
    length: int,
    down_sizes: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return source indices and weights for trilinear upsampling.

    Args:
        length: Original length along the degraded axis.
        down_sizes: Downsampled sizes per batch element.
        device: Device where the indices should be created.

    Returns:
        Lower source indices, upper source indices and upper weights,
            each shaped `(B, length)`.
    """
    positions = torch.arange(length, dtype=torch.float32, device=device)
    if length == 1:
        lowres_positions = torch.zeros(
            len(down_sizes),
            1,
            dtype=torch.float32,
            device=device,
        )
    else:
        scale = (down_sizes.to(torch.float32) - 1.0) / (length - 1)
        lowres_positions = positions * rearrange(scale, "b -> b 1")
    lower_lowres = lowres_positions.floor().to(torch.long)
    upper_lowres = torch.minimum(
        lower_lowres + 1,
        rearrange(down_sizes, "b -> b 1") - 1,
    )
    weights = lowres_positions - lower_lowres.to(torch.float32)
    lower = _downsample_source_indices(length, down_sizes, lower_lowres)
    upper = _downsample_source_indices(length, down_sizes, upper_lowres)
    return lower, upper, weights


def _downsample_source_indices(
    length: int,
    down_sizes: torch.Tensor,
    lowres_indices: torch.Tensor,
) -> torch.Tensor:
    """Map low-resolution indices to nearest-downsampled source indices.

    Args:
        length: Original length along the degraded axis.
        down_sizes: Downsampled sizes per batch element.
        lowres_indices: Low-resolution indices.

    Returns:
        Original source indices.
    """
    source = torch.div(
        lowres_indices * length,
        rearrange(down_sizes, "b -> b 1"),
        rounding_mode="floor",
    )
    return source.clamp(max=length - 1)


def _gather_axis(
    data: torch.Tensor,
    source_indices: torch.Tensor,
    axis: int,
) -> torch.Tensor:
    """Gather `data` along one spatial axis with per-element indices.

    Args:
        data: `(B, C, I, J, K)` tensor.
        source_indices: `(B, N)` indices for the selected spatial axis.
        axis: Spatial axis (0, 1, or 2) to gather.

    Returns:
        Gathered `(B, C, I, J, K)` tensor.
    """
    if axis == 0:
        indices = rearrange(source_indices, "b n -> b 1 n 1 1")
    elif axis == 1:
        indices = rearrange(source_indices, "b n -> b 1 1 n 1")
    else:
        indices = rearrange(source_indices, "b n -> b 1 1 1 n")
    indices = indices.expand_as(data)
    return torch.gather(data, dim=axis + 2, index=indices)


def _broadcast_axis_weights(weights: torch.Tensor, axis: int) -> torch.Tensor:
    """Broadcast interpolation weights over channels and untouched axes.

    Args:
        weights: `(B, N)` interpolation weights.
        axis: Spatial axis (0, 1, or 2) corresponding to `N`.

    Returns:
        Weights broadcast-compatible with `(B, C, I, J, K)`.
    """
    if axis == 0:
        return rearrange(weights, "b n -> b 1 n 1 1")
    if axis == 1:
        return rearrange(weights, "b n -> b 1 1 n 1")
    return rearrange(weights, "b n -> b 1 1 1 n")


def _simulate_anisotropy(
    data: torch.Tensor,
    *,
    axis: int,
    factor: float,
    mode: str,
) -> torch.Tensor:
    """Downsample then upsample one axis of a 5-D tensor.

    Args:
        data: `(B, C, I, J, K)` tensor.
        axis: Spatial axis (0, 1, or 2) to degrade.
        factor: Downsampling factor (> 1).
        mode: Interpolation mode for upsampling (`"nearest"` or
            `"linear"`).

    Returns:
        Degraded `(B, C, I, J, K)` tensor with original shape.
    """
    original_shape = list(data.shape[2:])
    down_shape = list(original_shape)
    down_shape[axis] = max(1, round(original_shape[axis] / factor))

    torch_mode_down = "nearest"
    torch_mode_up = "nearest" if mode == "nearest" else "trilinear"

    # Downsample.
    downsampled = functional.interpolate(
        data.float(),
        size=down_shape,
        mode=torch_mode_down,
    )
    # Upsample back.
    upsampled = functional.interpolate(
        downsampled,
        size=original_shape,
        mode=torch_mode_up,
        align_corners=None if torch_mode_up == "nearest" else True,
    )
    return upsampled.to(data.dtype)
