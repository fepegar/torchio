from typing import Union, Tuple, Optional
import numpy as np
import warnings
from deprecated import deprecated
from .pad import Pad
from .crop import Crop
from .bounds_transform import BoundsTransform, TypeShape, TypeSixBounds
from ....torchio import DATA
from ....utils import is_image_dict, check_consistent_shape, round_up


class CropOrPad(BoundsTransform):
    """Crop and/or pad an image to a target shape.

    This transform modifies the affine matrix associated to the volume so that
    physical positions of the voxels are maintained.

    Args:
        target_shape: Tuple :math:`(D, H, W)`. If a single value :math:`N` is
            provided, then :math:`D = H = W = N`.
        padding_mode: See :py:class:`~torchio.transforms.Pad`.
        padding_fill: Same as :attr:`fill` in
            :py:class:`~torchio.transforms.Pad`.
        mask_name: If ``None``, the centers of the input and output volumes
            will be the same.
            If a string is given, the output volume center will be the center
            of the bounding box of non-zero values in the image named
            :py:attr:`mask_name`.

    Example:
        >>> import torchio
        >>> from torchio.tranforms import CropOrPad
        >>> subject = torchio.Subject(
        ...     torchio.Image('chest_ct', 'subject_a_ct.nii.gz', torchio.INTENSITY),
        ...     torchio.Image('heart_mask', 'subject_a_heart_seg.nii.gz', torchio.LABEL),
        ... )
        >>> sample = torchio.ImagesDataset([subject])[0]
        >>> sample['chest_ct'][torchio.DATA].shape
        torch.Size([1, 512, 512, 289])
        >>> transform = CropOrPad(
        ...     (120, 80, 180),
        ...     mask_name='heart_mask',
        ... )
        >>> transformed = transform(sample)
        >>> transformed['chest_ct'][torchio.DATA].shape
        torch.Size([1, 120, 80, 180])
    """
    def __init__(
            self,
            target_shape: Union[int, TypeShape],
            padding_mode: str = 'constant',
            padding_fill: Optional[float] = None,
            mask_name: Optional[str] = None,
            ):
        super().__init__(target_shape)
        self.padding_mode = padding_mode
        self.padding_fill = padding_fill
        if mask_name is not None and not isinstance(mask_name, str):
            message = (
                'If mask_name is not None, it must be a string,'
                f' not {type(mask_name)}'
            )
            raise ValueError(message)
        self.mask_name = mask_name
        if self.mask_name is None:
            self.compute_crop_or_pad = self._compute_center_crop_or_pad
        else:
            if not isinstance(mask_name, str):
                message = (
                    'If mask_name is not None, it must be a string,'
                    f' not {type(mask_name)}'
                )
                raise ValueError(message)
            self.compute_crop_or_pad = self._compute_mask_center_crop_or_pad

    @staticmethod
    def _bbox_mask(
            mask_volume: np.ndarray,
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Return 6 coordinates of a 3D bounding box from a given mask.

        Taken from `this SO question <https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array>`_.

        Args:
            mask_volume: 3D NumPy array.
        """
        r = np.any(mask_volume, axis=(1, 2))
        c = np.any(mask_volume, axis=(0, 2))
        z = np.any(mask_volume, axis=(0, 1))
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]
        return np.array([rmin, cmin, zmin]), np.array([rmax, cmax, zmax])

    @staticmethod
    def _get_sample_shape(sample: dict) -> TypeShape:
        """Return the shape of the first image in the sample."""
        check_consistent_shape(sample)
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            data = image_dict[DATA].shape[1:]  # remove channels dimension
            break
        return data

    @staticmethod
    def _get_six_bounds_parameters(
            parameters: np.ndarray,
            ) -> TypeSixBounds:
        r"""Compute bounds parameters for ITK filters.

        Args:
            parameters: Tuple :math:`(d, h, w)` with the number of voxels to be
                cropped or padded.

        Returns:
            Tuple :math:`(d_{ini}, d_{fin}, h_{ini}, h_{fin}, w_{ini}, w_{fin})`,
            where :math:`n_{ini} = \left \lceil \frac{n}{2} \right \rceil` and
            :math:`n_{fin} = \left \lfloor \frac{n}{2} \right \rfloor`.

        Example:
            >>> p = np.array((4, 0, 7))
            >>> _get_six_bounds_parameters(p)
            (2, 2, 0, 0, 4, 3)

        """
        parameters = parameters / 2
        result = []
        for n in parameters:
            ini, fin = int(np.ceil(n)), int(np.floor(n))
            result.extend([ini, fin])
        return tuple(result)

    def _compute_cropping_padding_from_shapes(
            self,
            source_shape: TypeShape,
            target_shape: TypeShape,
            ) -> Tuple[Optional[TypeSixBounds], Optional[TypeSixBounds]]:
        diff_shape = target_shape - source_shape

        cropping = -np.minimum(diff_shape, 0)
        if cropping.any():
            cropping_params = self._get_six_bounds_parameters(cropping)
        else:
            cropping_params = None

        padding = np.maximum(diff_shape, 0)
        if padding.any():
            padding_params = self._get_six_bounds_parameters(padding)
        else:
            padding_params = None

        return padding_params, cropping_params

    def _compute_center_crop_or_pad(
            self,
            sample: dict,
            ) -> Tuple[Optional[TypeSixBounds], Optional[TypeSixBounds]]:
        source_shape = self._get_sample_shape(sample)
        # The parent class turns the 3-element shape tuple (d, h, w)
        # into a 6-element bounds tuple (d, d, h, h, w, w)
        target_shape = np.array(self.bounds_parameters[::2])
        parameters = self._compute_cropping_padding_from_shapes(
            source_shape, target_shape)
        padding_params, cropping_params = parameters
        return padding_params, cropping_params

    def _compute_mask_center_crop_or_pad(
            self,
            sample: dict,
            ) -> Tuple[Optional[TypeSixBounds], Optional[TypeSixBounds]]:
        if self.mask_name not in sample:
            message = (
                f'Mask name "{self.mask_name}"'
                f' not found in sample keys "{tuple(sample.keys())}".'
                ' Using volume center instead'
            )
            warnings.warn(message)
            return self._compute_center_crop_or_pad(sample=sample)

        mask = sample[self.mask_name][DATA].numpy()

        if not np.any(mask):
            message = (
                f'All values found in the mask "{self.mask_name}"'
                ' are zero. Using volume center instead'
            )
            warnings.warn(message)
            return self._compute_center_crop_or_pad(sample=sample)

        # Original sample shape (from mask shape)
        sample_shape = mask.shape[1:]  # remove channels dimension
        # Calculate bounding box of the mask center
        bb_min, bb_max = self._bbox_mask(mask[0])
        # Coordinates of the mask center
        center_mask = (bb_max - bb_min) / 2 + bb_min
        # List of padding to do
        padding = []
        # Final cropping (after padding)
        cropping = []
        for dim, center_dimension in enumerate(center_mask):
            # Compute coordinates of the target shape taken from the center of
            # the mask
            center_dim = round_up(center_dimension)
            begin = center_dim - (self.bounds_parameters[2 * dim] / 2)
            end = center_dim + (self.bounds_parameters[2 * dim + 1] / 2)
            # Check if dimension needs padding (before or after)
            begin_pad = round(abs(min(begin, 0)))
            end_pad = round(max(end - sample_shape[dim], 0))
            # Check if cropping is needed
            begin_crop = round(max(begin, 0))
            end_crop = abs(round(min(end - sample_shape[dim], 0)))
            # Add padding values of the dim to the list
            padding.append(begin_pad)
            padding.append(end_pad)
            # Add the slice of the dimension to take
            cropping.append(begin_crop)
            cropping.append(end_crop)
        # Conversion for SimpleITK compatibility
        padding = np.asarray(padding, dtype=int)
        cropping = np.asarray(cropping, dtype=int)
        padding_params = tuple(padding.tolist()) if padding.any() else None
        cropping_params = tuple(cropping.tolist()) if padding.any() else None
        return padding_params, cropping_params

    def apply_transform(self, sample: dict) -> dict:
        padding_params, cropping_params = self.compute_crop_or_pad(sample)
        padding_kwargs = dict(
            padding_mode=self.padding_mode, fill=self.padding_fill)
        if padding_params is not None:
            sample = Pad(padding_params, **padding_kwargs)(sample)
        if cropping_params is not None:
            sample = Crop(cropping_params)(sample)
        return sample


@deprecated('CenterCropOrPad is deprecated. Use CropOrPad instead.')
class CenterCropOrPad(CropOrPad):
    pass
