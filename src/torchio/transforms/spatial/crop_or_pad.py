"""CropOrPad transform: crop and/or pad to a target shape."""

from __future__ import annotations

import copy as _copy
import math
from typing import Any
from typing import Literal

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from ...data.affine import AffineMatrix
from ...data.backends import ImageDataBackend
from ...data.backends import normalize_index
from ...data.batch import SubjectsBatch
from ...data.image import Image
from ...data.subject import Subject
from ...types import SliceIndex
from ...types import TypeAffineMatrix
from ...types import TypeSixInts
from ...types import TypeSpacing
from ...types import TypeTensorShape
from ...types import TypeThreeInts
from ..compose import Compose
from ..transform import AppliedTransform
from ..transform import SpatialTransform
from ._padding import PaddingMode
from ._padding import pad_tensor
from ._padding import parse_padding_mode
from .crop import Crop
from .pad import Pad

#: Accepted target shape specifications.
#: `int` or `float` → same size for each axis.
#: 3-tuple → per axis; use `None` to leave an axis unchanged.
TargetShapeParam = (
    int | float | tuple[int | float | None, int | float | None, int | float | None]
)

#: Accepted unit values.
Units = Literal["voxels", "mm", "cm"]

#: Accepted crop location strategies.
Location = Literal["center", "random"]


def _parse_target_shape(
    target_shape: TargetShapeParam,
) -> tuple[float | None, float | None, float | None]:
    """Normalise target_shape to a 3-tuple of floats or None."""
    if isinstance(target_shape, (int, float)):
        return (float(target_shape), float(target_shape), float(target_shape))
    values = list(target_shape)
    n = len(values)
    if n == 3:
        a, b, c = values
        return (
            None if a is None else float(a),
            None if b is None else float(b),
            None if c is None else float(c),
        )
    msg = f"target_shape must have 1 or 3 values, got {n}"
    raise ValueError(msg)


def _to_voxels(
    target: tuple[float | None, float | None, float | None],
    units: Units,
    spacing: TypeSpacing,
    current_shape: TypeThreeInts,
) -> TypeThreeInts:
    """Convert a target shape from the given units to integer voxels.

    `None` entries are replaced with the current size along that axis.
    """
    result: list[int] = []
    for t, sp, cur in zip(target, spacing, current_shape, strict=True):
        if t is None:
            result.append(cur)
        elif units == "voxels":
            result.append(round(t))
        else:
            factor = 10.0 if units == "cm" else 1.0
            result.append(round(t * factor / sp))
    return (result[0], result[1], result[2])


def _split_per_axis(
    diff: int,
    location: Location,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute (pad_ini, pad_fin) and (crop_ini, crop_fin) for one axis."""
    if diff > 0:
        ini = math.ceil(diff / 2)
        fin = math.floor(diff / 2)
        return (ini, fin), (0, 0)
    if diff < 0:
        amount = -diff
        if location == "random":
            ini = int(torch.randint(0, amount + 1, (1,)).item())
        else:
            ini = math.ceil(amount / 2)
        return (0, 0), (ini, amount - ini)
    return (0, 0), (0, 0)


def _compute_crop_and_pad(
    current_shape: TypeThreeInts,
    target_shape: TypeThreeInts,
    *,
    only_crop: bool,
    only_pad: bool,
    location: Location = "center",
) -> tuple[TypeSixInts | None, TypeSixInts | None]:
    """Compute per-side crop and pad amounts to go from current to target.

    Args:
        location: `"center"` splits evenly; `"random"` picks a
            random crop start position for axes that need cropping.

    Returns:
        `(padding_six, cropping_six)`: either may be `None` when no
        padding or cropping is needed (or when `only_crop` / `only_pad`
        suppress it).
    """
    pad_values: list[int] = []
    crop_values: list[int] = []
    for cur, tgt in zip(current_shape, target_shape, strict=True):
        pad, crop = _split_per_axis(tgt - cur, location)
        pad_values.extend(pad)
        crop_values.extend(crop)

    has_padding = any(v > 0 for v in pad_values)
    has_cropping = any(v > 0 for v in crop_values)

    padding: TypeSixInts | None = None
    if has_padding and not only_crop:
        padding = (
            pad_values[0],
            pad_values[1],
            pad_values[2],
            pad_values[3],
            pad_values[4],
            pad_values[5],
        )

    cropping: TypeSixInts | None = None
    if has_cropping and not only_pad:
        cropping = (
            crop_values[0],
            crop_values[1],
            crop_values[2],
            crop_values[3],
            crop_values[4],
            crop_values[5],
        )

    return padding, cropping


class _CroppedBackend:
    """Backend wrapper that defers spatial cropping until data is accessed."""

    __slots__ = ("_affine", "_shape", "_source", "_spatial_slices")

    def __init__(
        self,
        source: ImageDataBackend,
        spatial_slices: tuple[slice, slice, slice],
        cropped_shape: tuple[int, int, int, int],
        affine: TypeAffineMatrix,
    ) -> None:
        self._source = source
        self._spatial_slices = spatial_slices
        self._shape = cropped_shape
        self._affine = affine

    @property
    def shape(self) -> TypeTensorShape:
        return self._shape

    @property
    def affine(self) -> TypeAffineMatrix:
        # The cropped affine (shifted origin), so it stays consistent with the
        # owning image's affine rather than the uncropped source affine.
        return self._affine

    @property
    def dtype(self) -> np.dtype:
        return self._source.dtype

    def to_tensor(self) -> Tensor:
        slices = (slice(None), *self._spatial_slices)
        return self._source[slices].to(dtype=torch.float32, copy=True)

    def __getitem__(self, slices: SliceIndex) -> Tensor:
        return self.to_tensor()[normalize_index(slices)]


class _PaddedBackend:
    """Backend wrapper that defers spatial padding until data is accessed."""

    __slots__ = ("_affine", "_fill", "_padding", "_padding_mode", "_shape", "_source")

    def __init__(
        self,
        source: ImageDataBackend,
        padding: TypeSixInts,
        padded_shape: tuple[int, int, int, int],
        affine: TypeAffineMatrix,
        padding_mode: PaddingMode = "constant",
        fill: float = 0,
    ) -> None:
        self._source = source
        self._padding = padding
        self._shape = padded_shape
        self._affine = affine
        self._padding_mode = padding_mode
        self._fill = fill

    @property
    def shape(self) -> TypeTensorShape:
        return self._shape

    @property
    def affine(self) -> TypeAffineMatrix:
        # The padded affine (shifted origin), so it stays consistent with the
        # owning image's affine rather than the unpadded source affine.
        return self._affine

    @property
    def dtype(self) -> np.dtype:
        return self._source.dtype

    def to_tensor(self) -> Tensor:
        base = self._source.to_tensor()
        return pad_tensor(
            base,
            self._padding,
            self._padding_mode,
            self._fill,
        )

    def __getitem__(self, slices: SliceIndex) -> Tensor:
        return self.to_tensor()[normalize_index(slices)]


def _get_images(
    subject: Subject,
    include: list[str] | None,
    exclude: list[str] | None,
) -> dict[str, Image]:
    """Filter subject images by include/exclude."""
    images = subject.images
    if include is not None:
        images = {k: v for k, v in images.items() if k in include}
    if exclude is not None:
        images = {k: v for k, v in images.items() if k not in exclude}
    return images


def _crop_image_lazy(image: Image, cropping: TypeSixInts) -> Image:
    """Crop an image lazily (data is only loaded when accessed)."""
    i0, i1, j0, j1, k0, k1 = cropping
    c, si, sj, sk = image.shape

    # `si - i1` is already `si` (equivalent to an open-ended stop) when
    # `i1 == 0`; `... or None` additionally replaced a stop of exactly 0
    # (cropping the whole axis, i.e. i1 == si) with None, silently leaving
    # that axis completely uncropped instead of empty.
    i_slice = slice(i0, si - i1)
    j_slice = slice(j0, sj - j1)
    k_slice = slice(k0, sk - k1)

    # Compute new affine
    affine_matrix = image.affine.data.clone()
    start_voxel = torch.tensor(
        [float(i0), float(j0), float(k0)],
        dtype=torch.float64,
    )
    affine_matrix[:3, 3] += affine_matrix[:3, :3] @ start_voxel
    new_affine = AffineMatrix(affine_matrix)

    if image.is_loaded:
        new_data = image.data[:, i_slice, j_slice, k_slice]
        return image.new_like(data=new_data, affine=new_affine)

    # Install a cropped backend on a new Image from the same source
    image._ensure_backend()
    if image._backend is not None and (
        image._path is not None or image._zarr_store is not None
    ):
        cropped_shape = (
            c,
            len(range(*i_slice.indices(si))),
            len(range(*j_slice.indices(sj))),
            len(range(*k_slice.indices(sk))),
        )
        source = image._path if image._path is not None else image._zarr_store
        new = type(image)(
            source,
            reader=image._reader,
            reader_kwargs=dict(image._reader_kwargs),
            affine=new_affine,
            **dict(image._metadata),
        )
        new._backend = _CroppedBackend(
            image._backend,
            (i_slice, j_slice, k_slice),
            cropped_shape,
            affine=new_affine.data,
        )
        return new

    # No backend (custom reader) → fall back to eager crop
    new_data = image.data[:, i_slice, j_slice, k_slice]
    return image.new_like(data=new_data, affine=new_affine)


def _pad_image_lazy(
    image: Image,
    padding: TypeSixInts,
    padding_mode: PaddingMode,
    fill: float,
) -> Image:
    """Pad an image lazily (data is only loaded when accessed)."""
    i0, i1, j0, j1, k0, k1 = padding
    c, si, sj, sk = image.shape

    # Compute new affine
    affine_matrix = image.affine.data.clone()
    origin_shift = affine_matrix[:3, :3] @ affine_matrix.new_tensor(
        [-float(i0), -float(j0), -float(k0)],
    )
    affine_matrix[:3, 3] += origin_shift
    new_affine = AffineMatrix(affine_matrix)

    padded_shape = (c, si + i0 + i1, sj + j0 + j1, sk + k0 + k1)

    if image.is_loaded:
        new_data = pad_tensor(
            image.data,
            padding,
            padding_mode,
            fill,
        )
        return image.new_like(data=new_data, affine=new_affine)

    # Install a padded backend on a new Image from the same source
    image._ensure_backend()
    if image._backend is not None and (
        image._path is not None or image._zarr_store is not None
    ):
        source = image._path if image._path is not None else image._zarr_store
        new = type(image)(
            source,
            reader=image._reader,
            reader_kwargs=dict(image._reader_kwargs),
            affine=new_affine,
            **dict(image._metadata),
        )
        new._backend = _PaddedBackend(
            image._backend,
            padding,
            padded_shape,
            affine=new_affine.data,
            padding_mode=padding_mode,
            fill=fill,
        )
        return new

    # No backend → fall back to eager pad
    new_data = pad_tensor(
        image.data,
        padding,
        padding_mode,
        fill,
    )
    return image.new_like(data=new_data, affine=new_affine)


class CropOrPad(SpatialTransform):
    r"""Crop and/or pad to a target spatial shape.

    If the current spatial size along an axis is larger than the target, that
    axis is cropped symmetrically from both sides. If it is smaller, it is
    padded symmetrically. The affine matrix is updated so that physical
    positions of the voxels are maintained.

    The target shape can be specified in voxels (the default), millimetres,
    or centimetres. When physical units are used, the target is converted to
    voxels at transform time using the image spacing.

    When the input is a `Subject` or `Image`, the transform operates
    lazily (data is not loaded from disk until it is actually accessed).

    Args:
        target_shape: Desired spatial shape. A single `int` broadcasts
            to all three axes. When `units` is `"mm"` or `"cm"`,
            values may be floats representing the physical extent along
            each axis. Use `None` for an axis to leave it unchanged,
            e.g., `(256, 256, None)`.
        units: Coordinate system for `target_shape`. One of
            `"voxels"` (default), `"mm"`, or `"cm"`.
        padding_mode: One of `'constant'`, `'reflect'`,
            `'replicate'`, `'circular'`, `'mean'`, `'median'`, or
            `'minimum'`. Statistical modes use one value computed from
            the whole image volume. For integer inputs, `'mean'` and
            `'median'` may be truncated to the input dtype and emit a
            warning.
        fill: Fill value when `padding_mode='constant'`.
        only_crop: If `True`, padding is never applied. Mutually
            exclusive with `only_pad`.
        only_pad: If `True`, cropping is never applied. Mutually
            exclusive with `only_crop`.
        location: Where to place the crop window when the image is
            larger than the target. `"center"` (default) centres the
            window; `"random"` picks a uniformly random position.
            Padding is always centred regardless of this parameter.
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.CropOrPad(target_shape=(120, 80, 180))
        >>> transform = tio.CropOrPad(target_shape=256)
        >>> transform = tio.CropOrPad(target_shape=(150.0, 200.0, 180.0), units='mm')
        >>> transform = tio.CropOrPad(target_shape=(15.0, 20.0, 18.0), units='cm')
        >>> transform = tio.CropOrPad(target_shape=256, only_pad=True)
        >>> transform = tio.CropOrPad(target_shape=(256, 256, None))  # keep depth
        >>> transform = tio.CropOrPad(target_shape=96, location='random')
        >>> transform = tio.CropOrPad(target_shape=256, padding_mode='mean')
    """

    def __init__(
        self,
        target_shape: TargetShapeParam,
        *,
        units: Units = "voxels",
        padding_mode: PaddingMode = "constant",
        fill: float = 0,
        only_crop: bool = False,
        only_pad: bool = False,
        location: Location = "center",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if only_crop and only_pad:
            msg = "only_crop and only_pad cannot both be True"
            raise ValueError(msg)
        if units not in ("voxels", "mm", "cm"):
            msg = f"units must be 'voxels', 'mm', or 'cm', got {units!r}"
            raise ValueError(msg)
        if location not in ("center", "random"):
            msg = f"location must be 'center' or 'random', got {location!r}"
            raise ValueError(msg)
        self.target_shape = _parse_target_shape(target_shape)
        self.units: Units = units
        self.padding_mode = parse_padding_mode(padding_mode)
        self.fill = fill
        self.only_crop = only_crop
        self.only_pad = only_pad
        self.location: Location = location

    def forward(self, data):
        """Apply the transform.

        For `Subject` and `Image` inputs, operates lazily per-image
        without loading data from disk. For batched inputs, falls back
        to the standard `SubjectsBatch` path.
        """
        if isinstance(data, (Subject, Image)):
            return self._forward_lazy(data)
        return super().forward(data)

    def _forward_lazy(self, data: Subject | Image) -> Subject | Image:
        is_image = isinstance(data, Image)
        if is_image:
            subject = Subject(tio_default_image=data)
        else:
            assert isinstance(data, Subject)
            subject = data

        if self.copy:
            subject = _copy.deepcopy(subject)

        if torch.rand(1).item() > self.p:
            return subject.tio_default_image if is_image else subject

        first_image = next(iter(subject.images.values()))
        current_shape: TypeThreeInts = first_image.spatial_shape
        target_voxels = _to_voxels(
            self.target_shape,
            self.units,
            first_image.affine.spacing,
            current_shape,
        )

        padding, cropping = _compute_crop_and_pad(
            current_shape,
            target_voxels,
            only_crop=self.only_crop,
            only_pad=self.only_pad,
            location=self.location,
        )

        self._apply_lazy_ops(subject, padding, cropping)

        return subject.tio_default_image if is_image else subject

    def _apply_lazy_ops(
        self,
        subject: Subject,
        padding: TypeSixInts | None,
        cropping: TypeSixInts | None,
    ) -> None:
        """Apply lazy pad/crop and record history."""
        images = _get_images(subject, self.include, self.exclude)
        include = None if self.include is None else list(self.include)
        exclude = None if self.exclude is None else list(self.exclude)

        if padding is not None:
            for name, image in images.items():
                subject._images[name] = _pad_image_lazy(
                    image,
                    padding,
                    self.padding_mode,
                    self.fill,
                )
            subject.applied_transforms.append(
                AppliedTransform(
                    name="Pad",
                    params={
                        "padding": padding,
                        "padding_mode": self.padding_mode,
                        "fill": self.fill,
                    },
                    include=include,
                    exclude=exclude,
                ),
            )

        if cropping is not None:
            images = _get_images(subject, self.include, self.exclude)
            for name, image in images.items():
                subject._images[name] = _crop_image_lazy(image, cropping)
            subject.applied_transforms.append(
                AppliedTransform(
                    name="Crop",
                    params={"cropping": cropping},
                    include=include,
                    exclude=exclude,
                ),
            )

        subject.applied_transforms.append(
            AppliedTransform(
                name="CropOrPad",
                params={"padding": padding, "cropping": cropping},
                include=include,
                exclude=exclude,
            ),
        )

    # --- Standard batch path (for SubjectsBatch, Tensor, etc.) ---

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        first_images = next(iter(batch.images.values()))
        spacing = first_images.affines[0].spacing

        data_tensor = first_images.data
        current_shape: TypeThreeInts = (
            data_tensor.shape[-3],
            data_tensor.shape[-2],
            data_tensor.shape[-1],
        )

        target_voxels = _to_voxels(
            self.target_shape,
            self.units,
            spacing,
            current_shape,
        )

        if self.units != "voxels":
            logger.debug(
                "CropOrPad target {} {} → {} voxels (spacing {} mm)",
                self.target_shape,
                self.units,
                target_voxels,
                spacing,
            )

        padding, cropping = _compute_crop_and_pad(
            current_shape,
            target_voxels,
            only_crop=self.only_crop,
            only_pad=self.only_pad,
            location=self.location,
        )

        return {"padding": padding, "cropping": cropping}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        padding: TypeSixInts | None = params["padding"]
        cropping: TypeSixInts | None = params["cropping"]

        transforms: list[SpatialTransform] = []
        if padding is not None:
            transforms.append(
                Pad(
                    padding=padding,
                    padding_mode=self.padding_mode,
                    fill=self.fill,
                    include=self.include,
                    exclude=self.exclude,
                )
            )
        if cropping is not None:
            transforms.append(
                Crop(
                    cropping=cropping,
                    include=self.include,
                    exclude=self.exclude,
                )
            )

        if transforms:
            pipeline = Compose(transforms, copy=False)
            batch = pipeline(batch)

        return batch
