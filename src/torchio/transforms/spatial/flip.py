"""Flip transform: reverse voxel order along spatial axes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from einops import rearrange

from ...data.batch import ImagesBatch
from ...data.batch import SubjectsBatch
from ..transform import SpatialTransform

# Map anatomical labels to axis indices.
# Only the first letter (uppercased) is used.
_LABEL_TO_AXIS: dict[str, tuple[str, str]] = {
    "L": ("L", "R"),
    "R": ("L", "R"),
    "A": ("A", "P"),
    "P": ("A", "P"),
    "I": ("I", "S"),
    "S": ("I", "S"),
}


def _resolve_axes(
    axes: int | str | Sequence[int | str],
    orientation: tuple[str, str, str] | None = None,
) -> tuple[int, ...]:
    """Normalise axes to a tuple of ints (0, 1, 2).

    Accepts ints, anatomical strings (`'L'`, `'Right'`, `'AP'`,
    etc.), or a mix. String axes are resolved using the image
    orientation.
    """
    if isinstance(axes, (int, str)):
        axes = (axes,)
    result: list[int] = []
    for axis in axes:
        if isinstance(axis, int):
            if axis not in (0, 1, 2):
                msg = f"Axis must be 0, 1, or 2; got {axis}"
                raise ValueError(msg)
            result.append(axis)
        elif isinstance(axis, str):
            letter = axis[0].upper()
            if letter not in _LABEL_TO_AXIS:
                msg = (
                    f"Unknown anatomical label {axis!r}."
                    " Use L, R, A, P, I, S or full names"
                    " like 'Left', 'Right', etc."
                )
                raise ValueError(msg)
            if orientation is None:
                msg = (
                    "Cannot resolve anatomical axis label"
                    f" {axis!r} without image orientation"
                )
                raise ValueError(msg)
            pair = _LABEL_TO_AXIS[letter]
            for dim, code in enumerate(orientation):
                if code in pair:
                    result.append(dim)
                    break
        else:
            msg = f"Axis must be int or str, got {type(axis).__name__}"
            raise TypeError(msg)
    return tuple(sorted(set(result)))


class Flip(SpatialTransform):
    r"""Reverse the order of elements in an image along the given axes.

    Args:
        axes: Index or tuple of indices of the spatial dimensions along
            which the image might be flipped. Integers must be in
            `{0, 1, 2}`. Anatomical labels may also be used, such as
            `'Left'`, `'Right'`, `'Anterior'`, `'Posterior'`,
            `'Inferior'`, `'Superior'`. Only the first letter of
            the string is used. Anatomical labels are resolved using
            the image orientation.
        flip_probability: Probability that each axis will be flipped
            (per-axis coin flip). This is independent of the `p`
            parameter, which gates the entire transform.
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Tip:
        Specifying the axes as anatomical labels is useful when the
        image orientation is not known.

    Examples:
        >>> import torchio as tio
        >>> # Flip along the first spatial axis
        >>> transform = tio.Flip(axes=0)
        >>> # Flip along the lateral axis (anatomical label)
        >>> transform = tio.Flip(axes='LR')
        >>> # Random per-axis flip with 50% chance each
        >>> transform = tio.Flip(axes=(0, 1, 2), flip_probability=0.5)
    """

    def __init__(
        self,
        *,
        axes: int | str | Sequence[int | str] = 0,
        flip_probability: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.axes = axes
        if not 0 <= flip_probability <= 1:
            msg = f"flip_probability must be in [0, 1], got {flip_probability}"
            raise ValueError(msg)
        self.flip_probability = flip_probability

    def make_params(
        self,
        batch: SubjectsBatch,
    ) -> dict[str, Any]:
        images = self._get_images(batch)
        if not images:
            return {"axes": ()}
        first_img = next(iter(images.values()))

        n = self._resolve_n(batch)
        if n is None:
            orientation = None
            if first_img.batch_size > 0:
                orientation = first_img[0].orientation
            resolved = _resolve_axes(self.axes, orientation)
            flip_mask = torch.rand(3) < self.flip_probability
            axes_to_flip = tuple(a for a in resolved if flip_mask[a].item())
            return {"axes": axes_to_flip}

        keep = self._keep_mask(batch, n)
        axes_list = self._sample_per_element_axes(n, first_img, keep)
        params = {"axes": axes_list}
        self._tag_batched(params, batch, n, keep, ["axes"])
        return params

    def _sample_per_element_axes(
        self,
        n: int,
        first_img: ImagesBatch,
        keep: torch.Tensor | None,
    ) -> list[list[int]]:
        """Sample the flip axes for each batch element.

        Gated-out elements (``keep[index]`` is false) get an empty axis
        list so they are left unflipped.

        Args:
            n: Number of batch elements.
            first_img: First selected image batch, used for per-element
                orientation.
            keep: Per-element keep mask, or ``None`` to keep all elements.

        Returns:
            One list of spatial axes (in ``{0, 1, 2}``) per element.
        """
        axes_list: list[list[int]] = []
        for index in range(n):
            if keep is not None and not keep[index]:
                axes_list.append([])
                continue
            # Resolve anatomical axes per element: each sample may have its
            # own orientation in a batch with per-sample affines.
            resolved = _resolve_axes(self.axes, first_img[index].orientation)
            flip_mask = torch.rand(3) < self.flip_probability
            axes_list.append([a for a in resolved if flip_mask[a].item()])
        return axes_list

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
        axes = params["axes"]
        if self._is_per_instance_params(params):
            for _name, img_batch in self._get_images(batch).items():
                shape = img_batch.data.shape[-3:]
                img_batch.data = _flip_per_element(img_batch.data, axes)
                for index, affine in enumerate(img_batch.affines):
                    _flip_affine(affine, axes[index], shape)
            return batch
        if not axes:
            return batch
        dims = [a - 3 for a in axes]
        for _name, img_batch in self._get_images(batch).items():
            shape = img_batch.data.shape[-3:]
            img_batch.data = torch.flip(img_batch.data, dims)
            for affine in img_batch.affines:
                _flip_affine(affine, axes, shape)
        return batch

    @property
    def invertible(self) -> bool:
        return True

    def inverse(self, params: dict[str, Any]) -> Flip | _FlipInverse:
        """Flip is its own inverse."""
        if self._is_per_instance_params(params):
            return _FlipInverse(axes_per_element=params["axes"], copy=False)
        return Flip(axes=params["axes"], copy=False)


def _flip_affine(
    affine: Any,
    axes: Sequence[int],
    shape: Sequence[int],
) -> None:
    """Update an `AffineMatrix` in place to match a data flip.

    Reversing the voxel order along an axis negates that axis's
    direction column and shifts the origin to the voxel that is now
    first along that axis, so world positions keep matching the data
    (mirroring how `Crop` and `Pad` update the affine's origin).

    Args:
        affine: The `AffineMatrix` to update in place.
        axes: Spatial axes (in `{0, 1, 2}`) that were flipped.
        shape: The image's `(I, J, K)` spatial shape.
    """
    if not axes:
        return
    matrix = affine._matrix
    rotation = matrix[:3, :3].clone()
    for axis in axes:
        matrix[:3, 3] += rotation[:, axis] * (shape[axis] - 1)
        matrix[:3, axis] = -matrix[:3, axis]


def _flip_per_element(
    data: torch.Tensor, axes_per_element: list[list[int]]
) -> torch.Tensor:
    """Flip each batch element along its own axes.

    Args:
        data: `(B, C, I, J, K)` tensor.
        axes_per_element: One list of spatial axes (in `{0, 1, 2}`) per
            element.

    Returns:
        The flipped `(B, C, I, J, K)` tensor.
    """
    batch_size = data.shape[0]
    result = data
    # Flip the whole batch along each spatial axis once, then select per
    # element with a boolean mask. Flips along distinct axes commute, so
    # composing them sequentially matches flipping an element's axes at once.
    for spatial_axis in range(3):
        flip_flags = torch.tensor(
            [spatial_axis in axes_per_element[index] for index in range(batch_size)],
            device=data.device,
        )
        if not bool(flip_flags.any()):
            continue
        flipped = torch.flip(result, [spatial_axis - 3])
        mask = rearrange(flip_flags, "b -> b 1 1 1 1")
        result = torch.where(mask, flipped, result)
    return result


class _FlipInverse(SpatialTransform):
    """Inverse of a per-instance [`Flip`][torchio.Flip] for history replay."""

    def __init__(
        self,
        *,
        axes_per_element: list[list[int]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._axes_per_element = axes_per_element

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        for _name, img_batch in self._get_images(batch).items():
            shape = img_batch.data.shape[-3:]
            img_batch.data = _flip_per_element(img_batch.data, self._axes_per_element)
            for index, affine in enumerate(img_batch.affines):
                _flip_affine(affine, self._axes_per_element[index], shape)
        return batch
