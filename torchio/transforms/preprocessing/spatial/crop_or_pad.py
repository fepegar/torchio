import warnings
from typing import Union, Tuple, Optional

import numpy as np

from .pad import Pad
from .crop import Crop
from .bounds_transform import BoundsTransform
from ...transform import TypeTripletInt, TypeSixBounds
from ....data.subject import Subject
from ....utils import round_up


class CropOrPad(BoundsTransform):
    """Crop and/or pad an image to a target shape.

    This transform modifies the affine matrix associated to the volume so that
    physical positions of the voxels are maintained.

    Args:
        target_shape: Tuple :math:`(W, H, D)`. If a single value :math:`N` is
            provided, then :math:`W = H = D = N`.
        padding_mode: Same as :attr:`padding_mode` in
            :class:`~torchio.transforms.Pad`.
        mask_name: If ``None``, the centers of the input and output volumes
            will be the same.
            If a string is given, the output volume center will be the center
            of the bounding box of non-zero values in the image named
            :attr:`mask_name`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> subject = tio.Subject(
        ...     chest_ct=tio.ScalarImage('subject_a_ct.nii.gz'),
        ...     heart_mask=tio.LabelMap('subject_a_heart_seg.nii.gz'),
        ... )
        >>> subject.chest_ct.shape
        torch.Size([1, 512, 512, 289])
        >>> transform = tio.CropOrPad(
        ...     (120, 80, 180),
        ...     mask_name='heart_mask',
        ... )
        >>> transformed = transform(subject)
        >>> transformed.chest_ct.shape
        torch.Size([1, 120, 80, 180])
    """
    def __init__(
            self,
            target_shape: Union[int, TypeTripletInt],
            padding_mode: Union[str, float] = 0,
            mask_name: Optional[str] = None,
            **kwargs
            ):
        super().__init__(target_shape, **kwargs)
        self.padding_mode = padding_mode
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
        """  # noqa: E501
        i_any = np.any(mask_volume, axis=(1, 2))
        j_any = np.any(mask_volume, axis=(0, 2))
        k_any = np.any(mask_volume, axis=(0, 1))
        i_min, i_max = np.where(i_any)[0][[0, -1]]
        j_min, j_max = np.where(j_any)[0][[0, -1]]
        k_min, k_max = np.where(k_any)[0][[0, -1]]
        bb_min = np.array([i_min, j_min, k_min])
        bb_max = np.array([i_max, j_max, k_max])
        return bb_min, bb_max

    @staticmethod
    def _get_six_bounds_parameters(
            parameters: np.ndarray,
            ) -> TypeSixBounds:
        r"""Compute bounds parameters for ITK filters.

        Args:
            parameters: Tuple :math:`(w, h, d)` with the number of voxels to be
                cropped or padded.

        Returns:
            Tuple :math:`(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})`,
            where :math:`n_{ini} = \left \lceil \frac{n}{2} \right \rceil` and
            :math:`n_{fin} = \left \lfloor \frac{n}{2} \right \rfloor`.

        Example:
            >>> p = np.array((4, 0, 7))
            >>> CropOrPad._get_six_bounds_parameters(p)
            (2, 2, 0, 0, 4, 3)
        """  # noqa: E501
        parameters = parameters / 2
        result = []
        for number in parameters:
            ini, fin = int(np.ceil(number)), int(np.floor(number))
            result.extend([ini, fin])
        return tuple(result)

    def _compute_cropping_padding_from_shapes(
            self,
            source_shape: TypeTripletInt,
            target_shape: TypeTripletInt,
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
            subject: Subject,
            ) -> Tuple[Optional[TypeSixBounds], Optional[TypeSixBounds]]:
        source_shape = subject.spatial_shape
        # The parent class turns the 3-element shape tuple (w, h, d)
        # into a 6-element bounds tuple (w, w, h, h, d, d)
        target_shape = np.array(self.bounds_parameters[::2])
        parameters = self._compute_cropping_padding_from_shapes(
            source_shape, target_shape)
        padding_params, cropping_params = parameters
        return padding_params, cropping_params

    def _compute_mask_center_crop_or_pad(
            self,
            subject: Subject,
            ) -> Tuple[Optional[TypeSixBounds], Optional[TypeSixBounds]]:
        if self.mask_name not in subject:
            message = (
                f'Mask name "{self.mask_name}"'
                f' not found in subject keys "{tuple(subject.keys())}".'
                ' Using volume center instead'
            )
            warnings.warn(message, RuntimeWarning)
            return self._compute_center_crop_or_pad(subject=subject)

        mask = subject[self.mask_name].numpy()

        if not np.any(mask):
            message = (
                f'All values found in the mask "{self.mask_name}"'
                ' are zero. Using volume center instead'
            )
            warnings.warn(message, RuntimeWarning)
            return self._compute_center_crop_or_pad(subject=subject)

        # Original subject shape (from mask shape)
        sample_shape = subject.spatial_shape
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
            begin_pad = round_up(abs(min(begin, 0)))
            end_pad = round(max(end - sample_shape[dim], 0))
            # Check if cropping is needed
            begin_crop = round_up(max(begin, 0))
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
        cropping_params = tuple(cropping.tolist()) if cropping.any() else None
        return padding_params, cropping_params

    def apply_transform(self, subject: Subject) -> Subject:
        padding_params, cropping_params = self.compute_crop_or_pad(subject)
        padding_kwargs = {'padding_mode': self.padding_mode}
        if padding_params is not None:
            subject = Pad(padding_params, **padding_kwargs)(subject)
        if cropping_params is not None:
            subject = Crop(cropping_params)(subject)
        return subject
