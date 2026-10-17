"""Normalize: linearly map voxel intensities to a target range."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any
from typing import cast

import torch
from torch import Tensor

from ...data.batch import ImagesBatch
from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from .._statistics import compute_quantile
from ..parameter_range import Choice
from ..parameter_range import _ParameterRange
from ..transform import IntensityTransform

TypeParameterValue = float | tuple | Choice | torch.distributions.Distribution


def _to_range(
    value: TypeParameterValue,
) -> _ParameterRange:
    """Convert a scalar, tuple, Choice, or Distribution to a _ParameterRange."""
    if isinstance(value, (torch.distributions.Distribution, Choice)):
        return _ParameterRange(value)
    if isinstance(value, (int, float)):
        return _ParameterRange(float(value))
    return _ParameterRange(tuple(float(v) for v in value))


class Normalize(IntensityTransform):
    r"""Linearly rescale voxel intensities to a target range.

    The transform clips values to an input range, then applies the
    affine map:

    $$v_{\text{out}} = \frac{v - m_{\min}}{m_{\max} - m_{\min}}
    \cdot (n_{\max} - n_{\min}) + n_{\min}$$

    All six numeric parameters are independently randomizable via
    scalar, `(low, high)` range, or `torch.distributions.Distribution`.

    Args:
        out_min: Lower bound of the output range.
        out_max: Upper bound of the output range.
        in_min: Lower bound of the input range. If `None`, determined
            from *percentile_low* of the (masked) input data.
        in_max: Upper bound of the input range. If `None`, determined
            from *percentile_high* of the (masked) input data.
        percentile_low: Lower percentile for auto input range.
        percentile_high: Upper percentile for auto input range.
            Use `(0.5, 99.5)` for the nn-UNet convention.
        masking_method: Which voxels to include when computing
            percentiles. `None` uses all voxels. A `str` is
            interpreted as a key to a
            [`LabelMap`][torchio.LabelMap] in the subject. A callable
            receives the image tensor and returns a boolean mask.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> # Rescale to [-1, 1] (default)
        >>> transform = tio.Normalize()
        >>> # CT windowing
        >>> transform = tio.Normalize(
        ...     out_min=0.0, out_max=1.0,
        ...     in_min=-1000.0, in_max=1000.0,
        ... )
        >>> # nn-UNet percentile clipping
        >>> transform = tio.Normalize(
        ...     percentile_low=0.5, percentile_high=99.5,
        ... )
        >>> # Random output range
        >>> transform = tio.Normalize(
        ...     out_min=(-1.0, 0.0), out_max=(0.5, 1.0),
        ... )
    """

    def __init__(
        self,
        *,
        out_min: TypeParameterValue = -1.0,
        out_max: TypeParameterValue = 1.0,
        in_min: TypeParameterValue | None = None,
        in_max: TypeParameterValue | None = None,
        percentile_low: TypeParameterValue = 0.0,
        percentile_high: TypeParameterValue = 100.0,
        masking_method: str | Callable[[Tensor], Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.out_min = _to_range(out_min)
        self.out_max = _to_range(out_max)
        self.in_min = _to_range(in_min) if in_min is not None else None
        self.in_max = _to_range(in_max) if in_max is not None else None
        self.percentile_low = _to_range(percentile_low)
        self.percentile_high = _to_range(percentile_high)
        self.masking_method = masking_method

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample random parameters and compute the input range.

        When per-instance augmentation is active, the output range is
        sampled independently per batch element; the data-driven input
        range stays batch-shared.

        Returns:
            Dict with `out_min`, `out_max`, and either `in_min`/`in_max`
            or `in_ranges` (per image name).
        """
        n = self._resolve_n(batch)
        out_min = self.out_min.sample_1d(n)
        out_max = self.out_max.sample_1d(n)
        pct_low = self.percentile_low.sample_1d()
        pct_high = self.percentile_high.sample_1d()

        params: dict[str, Any] = {
            "out_min": self._serialize_param(out_min),
            "out_max": self._serialize_param(out_max),
        }
        # If explicit in_min/in_max are given, sample them directly.
        if self.in_min is not None and self.in_max is not None:
            params["in_min"] = self.in_min.sample_1d()
            params["in_max"] = self.in_max.sample_1d()
        else:
            # Otherwise, compute per-image input range from percentiles.
            in_ranges: dict[str, list[tuple[float, float]]] = {}
            for name, img_batch in self._get_images(batch).items():
                in_ranges[name] = [
                    _percentile_range(
                        img_batch.data[index],
                        self._get_mask(img_batch, batch, index),
                        pct_low,
                        pct_high,
                        name,
                    )
                    for index in range(len(img_batch.data))
                ]
            params["in_ranges"] = in_ranges

        if n is not None:
            self._tag_batched(params, batch, n, None, ["out_min", "out_max"])
        return params

    @property
    def supports_per_instance_params(self) -> bool:
        return True

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Clip and linearly rescale each selected image."""
        for name, img_batch in self._get_images(batch).items():
            data = img_batch.data.float()
            if "in_min" in params:
                ranges = [(params["in_min"], params["in_max"])] * len(data)
            else:
                in_ranges = params.get("in_ranges", {})
                if name not in in_ranges:
                    continue
                ranges = in_ranges[name]

            in_min, in_max = _broadcast_ranges(ranges, data)
            in_range = in_max - in_min
            degenerate = in_range == 0
            if bool(degenerate.any()):
                flat = degenerate.flatten().nonzero().flatten().tolist()
                warnings.warn(
                    f'Cannot rescale "{name}": input range is zero'
                    f" for sample(s) {flat}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if bool(degenerate.all()):
                    continue

            out_min, out_range = _out_min_and_range(
                params["out_min"],
                params["out_max"],
                data,
            )
            safe_range = torch.where(degenerate, torch.ones_like(in_range), in_range)
            rescaled = data.clamp(in_min, in_max)
            rescaled = (rescaled - in_min) / safe_range * out_range + out_min
            # Samples with a zero-width input range are left untouched
            img_batch.data = torch.where(degenerate, data, rescaled)

        return batch

    @property
    def invertible(self) -> bool:
        """Whether this transform can be inverted."""
        return True

    def inverse(self, params: dict[str, Any]) -> _RescaleInverse:
        """Build the inverse transform from recorded parameters."""
        return _RescaleInverse(
            out_min=params["out_min"],
            out_max=params["out_max"],
            in_min=params.get("in_min"),
            in_max=params.get("in_max"),
            in_ranges=params.get("in_ranges"),
            copy=False,
        )

    def _get_mask(
        self,
        img_batch: ImagesBatch,
        batch: SubjectsBatch,
        index: int = 0,
    ) -> Tensor | None:
        """Resolve masking_method to a boolean tensor or None."""
        if self.masking_method is None:
            return None
        if callable(self.masking_method) and not isinstance(self.masking_method, str):
            return self.masking_method(img_batch.data[index]).bool()
        # String key: look up a LabelMap in the batch.
        if isinstance(self.masking_method, str):
            key = self.masking_method
            if key not in batch.images:
                msg = (
                    f'Masking method "{key}" not found in batch images.'
                    f" Available: {list(batch.images.keys())}"
                )
                raise KeyError(msg)
            mask_batch = batch.images[key]
            if not issubclass(mask_batch._image_class, LabelMap):
                msg = f'Masking method "{key}" must refer to a LabelMap.'
                raise TypeError(msg)
            return mask_batch.data[index].bool()
        msg = (
            "masking_method must be None, str, or callable, got"
            f" {type(self.masking_method)}"
        )
        raise TypeError(msg)


class _RescaleInverse(IntensityTransform):
    """Inverse of Normalize for history replay."""

    def __init__(
        self,
        *,
        out_min: float | list[float],
        out_max: float | list[float],
        in_min: float | None,
        in_max: float | None,
        in_ranges: dict[str, list[tuple[float, float]]] | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._out_min = out_min
        self._out_max = out_max
        self._in_min = in_min
        self._in_max = in_max
        self._in_ranges = in_ranges

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Return empty params; all state is in instance attributes."""
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Reverse the linear rescaling."""
        for name, img_batch in self._get_images(batch).items():
            data = img_batch.data.float()
            if self._in_min is not None and self._in_max is not None:
                ranges = [(self._in_min, self._in_max)] * len(data)
            elif self._in_ranges is not None and name in self._in_ranges:
                ranges = self._in_ranges[name]
            else:
                continue

            in_min, in_max = _broadcast_ranges(ranges, data)
            in_range = in_max - in_min
            degenerate = in_range == 0
            if bool(degenerate.all()):
                continue
            in_range = torch.where(degenerate, torch.ones_like(in_range), in_range)

            out_min, out_range = _out_min_and_range(
                self._out_min,
                self._out_max,
                data,
            )
            if isinstance(out_range, float):
                if out_range == 0:
                    continue
                reversed_data = (data - out_min) / out_range * in_range + in_min
                data = torch.where(degenerate, data, reversed_data)
            else:
                # Per-element: leave zero-range elements (out_min == out_max)
                # unchanged instead of dividing by zero.
                out_range_t = cast("Tensor", out_range)
                out_min_t = cast("Tensor", out_min)
                zero = out_range_t == 0
                safe_range = torch.where(
                    zero, torch.ones_like(out_range_t), out_range_t
                )
                reversed_data = (data - out_min_t) / safe_range * in_range + in_min
                data = torch.where(zero | degenerate, data, reversed_data)
            img_batch.data = data

        return batch


def _broadcast_ranges(
    ranges: list[tuple[float, float]],
    data: Tensor,
) -> tuple[Tensor, Tensor]:
    """Turn one (min, max) pair per sample into tensors that broadcast over `data`.

    Args:
        ranges: One `(min, max)` pair per sample in the batch.
        data: The batched tensor the ranges will be applied to.

    Returns:
        A `(min, max)` pair of tensors shaped `(B, 1, ..., 1)`.
    """
    shape = (len(ranges),) + (1,) * (data.ndim - 1)
    kwargs = {"dtype": data.dtype, "device": data.device}
    mins = torch.tensor([low for low, _ in ranges], **kwargs).reshape(shape)
    maxs = torch.tensor([high for _, high in ranges], **kwargs).reshape(shape)
    return mins, maxs


def _out_min_and_range(
    out_min: float | list[float],
    out_max: float | list[float],
    data: Tensor,
) -> tuple[float | Tensor, float | Tensor]:
    """Resolve the output min and range, broadcasting per element if needed.

    Args:
        out_min: Scalar (batch-shared) or per-element output minimum.
        out_max: Scalar or per-element output maximum.
        data: The `(B, C, I, J, K)` tensor being rescaled.

    Returns:
        A `(out_min, out_range)` pair, each a float (scalar case) or a
        `(B, 1, 1, 1, 1)` tensor (per-element case).
    """
    if isinstance(out_min, list):
        from einops import rearrange

        min_t = torch.tensor(out_min, dtype=torch.float32, device=data.device)
        max_t = torch.tensor(out_max, dtype=torch.float32, device=data.device)
        min_b = rearrange(min_t, "b -> b 1 1 1 1")
        max_b = rearrange(max_t, "b -> b 1 1 1 1")
        return min_b, max_b - min_b
    low = out_min
    high = cast("float", out_max)
    return low, high - low


def _percentile_range(
    tensor: Tensor,
    mask: Tensor | None,
    pct_low: float,
    pct_high: float,
    image_name: str,
) -> tuple[float, float]:
    """Compute the input range from percentiles of (masked) data.

    Args:
        tensor: `(C, I, J, K)` image tensor (first sample).
        mask: Optional boolean mask with compatible shape, or `None`.
        pct_low: Lower percentile (0-100).
        pct_high: Upper percentile (0-100).
        image_name: Used in warning messages.

    Returns:
        `(in_min, in_max)` tuple.
    """
    values = tensor[mask.expand_as(tensor)] if mask is not None else tensor.reshape(-1)

    if values.numel() == 0:
        warnings.warn(
            f'Cannot compute percentiles for "{image_name}": mask is empty.'
            " Using full range.",
            RuntimeWarning,
            stacklevel=3,
        )
        values = tensor.reshape(-1)

    low = float(compute_quantile(values.float(), pct_low / 100.0).item())
    high = float(compute_quantile(values.float(), pct_high / 100.0).item())
    return low, high


# Backwards-compatible alias.
RescaleIntensity = Normalize
