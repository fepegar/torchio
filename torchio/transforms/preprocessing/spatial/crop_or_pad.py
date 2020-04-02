from typing import Union, Tuple, Optional
import numpy as np
from .pad import Pad
from .crop import Crop
from .bounds_transform import BoundsTransform
from ....torchio import DATA
from ....utils import is_image_dict, check_consistent_shape


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
        mode: Whether to crop/pad using the image center or the center of the
            bounding box with non-zero values of a given mask with name
            :py:attr:`mask_name`.
            Possible values are ``'center'`` or ``'mask'``.
        mask_name: If :py:attr:`mode` is ``'mask'``, name of the mask from
            which to extract the bounding box.

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
        ...     padding_mode='reflect',
        ...     mode='mask',
        ...     mask_name='heart_mask',
        ... )
        >>> transformed = transform(sample)
        >>> transformed['chest_ct'][torchio.DATA].shape
        torch.Size([1, 120, 80, 180])
    """
    def __init__(
            self,
            target_shape: Union[int, Tuple[int, int, int]],
            padding_mode: str = 'constant',
            padding_fill: Optional[float] = None,
            mode: str = 'center',
            mask_name: Optional[str] = None,
            ):
        super().__init__(target_shape)
        self.mode = mode
        self.padding_mode = padding_mode
        self.padding_fill = padding_fill
        if mode not in {'center', 'mask'}:
            message = f'Mode must be "center" or "mask", not "{mode}"'
            raise ValueError(message)
        if mode == 'mask':
            if mask_name is None:
                message = 'If mode is "mask", mask_name cannot be None'
                raise ValueError(message)
            self.mask_name = mask_name
            self.compute_crop_or_pad = self._compute_mask_center_crop_or_pad
        else:
            self.compute_crop_or_pad = self._compute_center_crop_or_pad

    @staticmethod
    def _bbox_mask(mask_volume: np.ndarray):
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
    def _get_sample_shape(sample: dict) -> Tuple[int]:
        """Return the shape of the first image in the sample."""
        check_consistent_shape(sample)
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            data = image_dict[DATA].shape[1:]  # remove channels dimension
            break
        return data

    @staticmethod
    def _get_six_bounds_parameters(parameters: np.ndarray):
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

    def _compute_center_crop_or_pad(self, sample: dict):
        source_shape = self._get_sample_shape(sample)
        # The parent class turns the 3-element shape tuple (d, h, w)
        # into a 6-element bounds tuple (d, d, h, h, w, w)
        target_shape = np.array(self.bounds_parameters[::2])
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

    def _compute_mask_center_crop_or_pad(self, sample: dict):
        if self.mask_name not in sample:
            message = (
                f'Mask name "{self.mask_name}"'
                f' not found in sample keys: {tuple(sample.keys())}'
            )
            raise KeyError(message)
        mask = sample[self.mask_name][DATA].numpy()
        # Original sample shape (from mask shape)
        sample_shape = np.squeeze(mask).shape
        # Calculate bounding box of the mask center
        bb_min, bb_max = self._bbox_mask(mask[0])
        # Coordinates of the mask center
        center_mask = (bb_max - bb_min) / 2 + bb_min
        # List of padding to do
        padding = []
        # Final cropping (after padding)
        cropping = []
        for dim, center_dim in enumerate(center_mask):
            # Compute coordinates of the target shape taken from the center of
            # the mask
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
        padding_params = np.asarray(padding, dtype=np.uint).tolist()
        cropping_params = np.asarray(cropping, dtype=np.uint).tolist()
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
