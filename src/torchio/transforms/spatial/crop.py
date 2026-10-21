"""Crop transform: remove voxels from the borders."""

from __future__ import annotations

from typing import Any

from ...data.batch import SubjectsBatch
from ...types import TypeSixInts
from ...types import TypeThreeInts
from ..transform import SpatialTransform

#: Accepted cropping specifications.
#: `int` → same amount from each side of each axis.
#: 3-tuple → symmetric per axis `(i, j, k)`.
#: 6-tuple → per-side `(i_ini, i_fin, j_ini, j_fin, k_ini, k_fin)`.
CroppingParam = int | TypeThreeInts | TypeSixInts


def _parse_cropping(cropping: CroppingParam) -> TypeSixInts:
    """Normalise cropping to a 6-tuple (i_ini, i_fin, j_ini, j_fin, k_ini, k_fin)."""
    if isinstance(cropping, int):
        return (cropping, cropping, cropping, cropping, cropping, cropping)
    values = list(cropping)
    n = len(values)
    if n == 3:
        i, j, k = values
        return (i, i, j, j, k, k)
    if n == 6:
        return (values[0], values[1], values[2], values[3], values[4], values[5])
    msg = f"Cropping must have 1, 3, or 6 values, got {n}"
    raise ValueError(msg)


class Crop(SpatialTransform):
    r"""Remove a border of voxels from each side of the volume.

    Args:
        cropping: Tuple
            $(i_\text{ini}, i_\text{fin}, j_\text{ini}, j_\text{fin},
            k_\text{ini}, k_\text{fin})$
            defining the number of voxels cropped from the edges of
            each axis. If the initial shape of the image is
            $I \times J \times K$, the final shape will be
            $(I - i_\text{ini} - i_\text{fin}) \times
            (J - j_\text{ini} - j_\text{fin}) \times
            (K - k_\text{ini} - k_\text{fin})$.
            If only three values $(i, j, k)$ are provided, then
            $i_\text{ini} = i_\text{fin} = i$,
            $j_\text{ini} = j_\text{fin} = j$ and
            $k_\text{ini} = k_\text{fin} = k$.
            If only one value $n$ is provided, then
            $i_\text{ini} = i_\text{fin} = j_\text{ini} =
            j_\text{fin} = k_\text{ini} = k_\text{fin} = n$.
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Crop(cropping=10)
        >>> # Equivalent to
        >>> transform = tio.Crop(cropping=(10, 10, 10))
        >>> # Equivalent to
        >>> transform = tio.Crop(cropping=(10, 10, 10, 10, 10, 10))
    """

    def __init__(
        self,
        *,
        cropping: CroppingParam,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cropping = _parse_cropping(cropping)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        return {"cropping": self.cropping}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        i0, i1, j0, j1, k0, k1 = params["cropping"]
        for _name, img_batch in self._get_images(batch).items():
            data = img_batch.data
            # data shape: (B, C, I, J, K)
            si = data.shape[-3]
            sj = data.shape[-2]
            sk = data.shape[-1]
            # `si - i1` is already `si` (equivalent to an open-ended stop) when
            # `i1 == 0`; `... or None` additionally replaced a stop of exactly
            # 0 (cropping the whole axis, i.e. i1 == si) with None, silently
            # leaving that axis completely uncropped instead of empty.
            img_batch.data = data[
                ...,
                i0 : si - i1,
                j0 : sj - j1,
                k0 : sk - k1,
            ]
            # Update each affine's origin
            for affine in img_batch.affines:
                origin_shift = affine.data[:3, :3] @ affine.data.new_tensor(
                    [float(i0), float(j0), float(k0)],
                )
                affine._matrix[:3, 3] += origin_shift
        return batch

    @property
    def invertible(self) -> bool:
        return True

    def inverse(self, params: dict[str, Any]) -> Any:
        """Inverse of Crop is Pad."""
        from .pad import Pad

        return Pad(padding=params["cropping"], copy=False)
