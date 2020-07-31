from pathlib import Path
from numbers import Number
from typing import Union, Tuple, Optional

import torch
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to

from ....data.subject import Subject
from ....data.image import Image
from ....torchio import DATA, AFFINE, TYPE, INTENSITY, TypeData
from ... import Transform, Interpolation



TypeSpacing = Union[float, Tuple[float, float, float]]
TypeTarget = Tuple[
    Optional[Union[Image, str]],
    Optional[Tuple[float, float, float]],
]


class Resample(Transform):
    """Change voxel spacing by resampling.

    Args:
        target: Tuple :math:`(s_d, s_h, s_w)`. If only one value
            :math:`n` is specified, then :math:`s_d = s_h = s_w = n`.
            If a string or :py:class:`~pathlib.Path` is given,
            all images will be resampled using the image
            with that name as reference or found at the path.
        pre_affine_name: Name of the *image key* (not subject key) storing an
            affine matrix that will be applied to the image header before
            resampling. If ``None``, the image is resampled with an identity
            transform. See usage in the example below.
        image_interpolation: String that defines the interpolation technique.
            Supported interpolation techniques for resampling
            are ``'nearest'``, ``'linear'`` and ``'bspline'``.
            Using a member of :py:class:`torchio.Interpolation` is still
            supported for backward compatibility,
            but will be removed in a future version.
        p: Probability that this transform will be applied.

    .. note:: Resampling is performed using
        :py:meth:`nibabel.processing.resample_to_output` or
        :py:meth:`nibabel.processing.resample_from_to`, depending on whether
        the target is a spacing or a reference image.

    Example:
        >>> import torchio
        >>> from torchio import Resample
        >>> from torchio.datasets import Colin27, FPG
        >>> transform = Resample(1)                     # resample all images to 1mm iso
        >>> transform = Resample((2, 2, 2))             # resample all images to 2mm iso
        >>> transform = Resample('t1')                  # resample all images to 't1' image space
        >>> colin = Colin27()  # this images are in the MNI space
        >>> fpg = FPG()  # matrices to the MNI space are included here
        >>> # Resample all images into the MNI space
        >>> transform = Resample(colin.t1.path, pre_affine_name='affine_matrix')
        >>> transformed = transform(fpg)  # images in fpg are now in MNI space
    """
    def __init__(
            self,
            target: Union[TypeSpacing, str, Path],
            image_interpolation: str = 'linear',
            pre_affine_name: Optional[str] = None,
            p: float = 1,
            copy: bool = True,
            ):
        super().__init__(p=p, copy=copy)
        self.reference_image, self.target_spacing = self.parse_target(target)
        self.interpolation_order = self.parse_interpolation(image_interpolation)
        self.affine_name = pre_affine_name

    def parse_target(
            self,
            target: Union[TypeSpacing, str],
            ) -> TypeTarget:
        if isinstance(target, (str, Path)):
            if Path(target).is_file():
                path = target
                image = Image(path)
                reference_image = image.data, image.affine
            else:
                reference_image = target
            target_spacing = None
        else:
            reference_image = None
            target_spacing = self.parse_spacing(target)
        return reference_image, target_spacing

    @staticmethod
    def parse_spacing(spacing: TypeSpacing) -> Tuple[float, float, float]:
        if isinstance(spacing, tuple) and len(spacing) == 3:
            result = spacing
        elif isinstance(spacing, Number):
            result = 3 * (spacing,)
        else:
            message = (
                'Target must be a string, a positive number'
                f' or a tuple of positive numbers, not {type(spacing)}'
            )
            raise ValueError(message)
        if np.any(np.array(spacing) <= 0):
            raise ValueError(f'Spacing must be positive, not "{spacing}"')
        return result

    def parse_interpolation(self, interpolation: str) -> int:
        interpolation = super().parse_interpolation(interpolation)

        if interpolation in (Interpolation.NEAREST, 'nearest'):
            order = 0
        elif interpolation in (Interpolation.LINEAR, 'linear'):
            order = 1
        elif interpolation in (Interpolation.BSPLINE, 'bspline'):
            order = 3
        else:
            message = f'Interpolation not implemented yet: {interpolation}'
            raise NotImplementedError(message)
        return order

    @staticmethod
    def check_affine(affine_name: str, image_dict: dict):
        if not isinstance(affine_name, str):
            message = (
                'Affine name argument must be a string,'
                f' not {type(affine_name)}'
            )
            raise TypeError(message)
        if affine_name in image_dict:
            matrix = image_dict[affine_name]
            if not isinstance(matrix, (np.ndarray, torch.Tensor)):
                message = (
                    'The affine matrix must be a NumPy array or PyTorch tensor,'
                    f' not {type(matrix)}'
                )
                raise TypeError(message)
            if matrix.shape != (4, 4):
                message = (
                    'The affine matrix shape must be (4, 4),'
                    f' not {matrix.shape}'
                )
                raise ValueError(message)

    @staticmethod
    def check_affine_key_presence(affine_name: str, sample: Subject):
        for image_dict in sample.get_images(intensity_only=False):
            if affine_name in image_dict:
                return
        message = (
            f'An affine name was given ("{affine_name}"), but it was not found'
            ' in any image in the sample'
        )
        raise ValueError(message)

    def apply_transform(self, sample: Subject) -> dict:
        use_reference = self.reference_image is not None
        use_pre_affine = self.affine_name is not None
        if use_pre_affine:
            self.check_affine_key_presence(self.affine_name, sample)
        images_dict = sample.get_images_dict(intensity_only=False).items()
        for image_name, image in images_dict:
            # Do not resample the reference image if there is one
            if use_reference and image_name == self.reference_image:
                continue

            # Choose interpolator
            if image[TYPE] != INTENSITY:
                interpolation_order = 0  # nearest neighbor
            else:
                interpolation_order = self.interpolation_order

            # Apply given affine matrix if found in image
            if use_pre_affine and self.affine_name in image:
                self.check_affine(self.affine_name, image)
                matrix = image[self.affine_name]
                if isinstance(matrix, torch.Tensor):
                    matrix = matrix.numpy()
                image[AFFINE] = matrix @ image[AFFINE]

            # Resample
            if use_reference:
                if isinstance(self.reference_image, str):
                    try:
                        ref_image = sample[self.reference_image]
                    except KeyError as error:
                        message = (
                            f'Reference name "{self.reference_image}"'
                            ' not found in sample'
                        )
                        raise ValueError(message) from error
                    reference = ref_image[DATA], ref_image[AFFINE]
                else:
                    reference = self.reference_image
                kwargs = dict(reference=reference)
            else:
                kwargs = dict(target_spacing=self.target_spacing)
            image[DATA], image[AFFINE] = self.apply_resample(
                image[DATA],
                image[AFFINE],
                interpolation_order,
                **kwargs,
            )
        return sample

    @staticmethod
    def apply_resample(
            tensor: torch.Tensor,  # (C, D, H, W)
            affine: np.ndarray,
            interpolation_order: int,
            target_spacing: Optional[Tuple[float, float, float]] = None,
            reference: Optional[Tuple[torch.Tensor, np.ndarray]] = None,
            ) -> Tuple[torch.Tensor, np.ndarray]:
        array = tensor.numpy()
        niis = []
        arrays_resampled = []
        if reference is None:
            for channel in array:
                nii = resample_to_output(
                    nib.Nifti1Image(channel, affine),
                    voxel_sizes=target_spacing,
                    order=interpolation_order,
                )
                arrays_resampled.append(nii.get_fdata(dtype=np.float32))
        else:
            reference_tensor, reference_affine = reference
            reference_array = reference_tensor.numpy()[0]
            for channel_array in array:
                nii = resample_from_to(
                    nib.Nifti1Image(channel_array, affine),
                    nib.Nifti1Image(reference_array, reference_affine),
                    order=interpolation_order,
                )
                arrays_resampled.append(nii.get_fdata(dtype=np.float32))
        tensor = torch.Tensor(arrays_resampled)
        return tensor, nii.affine

    @staticmethod
    def get_sigma(downsampling_factor, spacing):
        """Compute optimal standard deviation for Gaussian kernel.

        From Cardoso et al., "Scale factor point spread function matching:
        beyond aliasing in image resampling", MICCAI 2015
        """
        k = downsampling_factor
        variance = (k ** 2 - 1 ** 2) * (2 * np.sqrt(2 * np.log(2))) ** (-2)
        sigma = spacing * np.sqrt(variance)
        return sigma
