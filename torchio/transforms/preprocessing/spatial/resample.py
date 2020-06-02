from numbers import Number
from typing import Union, Tuple, Optional
from pathlib import Path

import torch
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to

from ....data.subject import Subject
from ....data.image import Image
from ....torchio import DATA, AFFINE, TYPE, INTENSITY
from ... import Interpolation
from ... import Transform


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
            are 'nearest','linear' and 'bspline'.
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
        >>> from torchio.transforms import Resample
        >>> from pathlib import Path
        >>> transform = Resample(1)                     # resample all images to 1mm iso
        >>> transform = Resample((1, 1, 1))             # resample all images to 1mm iso
        >>> transform = Resample('t1')                  # resample all images to 't1' image space
        >>> transform = Resample('path/to/ref.nii.gz')  # resample all images to space of image at this path
        >>>
        >>> # Affine matrices are added to each image
        >>> matrix_to_mni = some_4_by_4_array  # e.g. result of registration to MNI space
        >>> subject = torchio.Subject(
        ...     t1=Image('t1.nii.gz', torchio.INTENSITY, to_mni=matrix_to_mni),
        ...     mni=Image('mni_152_lin.nii.gz', torchio.INTENSITY),
        ... )
        >>> resample = Resample(
        ...     'mni',  # this is a subject key
        ...     affine_name='to_mni',  # this is an image key
        ... )
        >>> dataset = torchio.ImagesDataset([subject], transform=resample)
        >>> sample = dataset[0]  # sample['t1'] is now in MNI space
    """
    def __init__(
            self,
            target: Union[TypeSpacing, str, Path],
            image_interpolation: str = 'linear',
            pre_affine_name: Optional[str] = None,
            p: float = 1,
            ):
        super().__init__(p=p)
        self.reference_image, self.target_spacing = self.parse_target(target)
        self.interpolation_order = self.parse_interpolation(image_interpolation)
        self.affine_name = pre_affine_name

    def parse_target(
            self,
            target: Union[TypeSpacing, str],
            ) -> TypeTarget:
        if isinstance(target, (str, Path)):
            if Path(target).is_file():
                reference_image = Image(target, INTENSITY).load()
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
            if not isinstance(matrix, np.ndarray):
                message = (
                    'The affine matrix must be a NumPy array,'
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
        for image_name, image_dict in images_dict:
            # Do not resample the reference image if there is one
            if use_reference and image_name == self.reference_image:
                continue

            # Choose interpolator
            if image_dict[TYPE] != INTENSITY:
                interpolation_order = 0  # nearest neighbor
            else:
                interpolation_order = self.interpolation_order

            # Apply given affine matrix if found in image
            if use_pre_affine and self.affine_name in image_dict:
                self.check_affine(self.affine_name, image_dict)
                matrix = image_dict[self.affine_name]
                image_dict[AFFINE] = matrix @ image_dict[AFFINE]

            # Resample
            args = image_dict[DATA], image_dict[AFFINE], interpolation_order
            if use_reference:
                if isinstance(self.reference_image, str):
                    try:
                        ref_image_dict = sample[self.reference_image]
                    except KeyError as error:
                        message = (
                            f'Reference name "{self.reference_image}"'
                            ' not found in sample'
                        )
                        raise ValueError(message) from error
                    reference = ref_image_dict[DATA], ref_image_dict[AFFINE]
                else:
                    reference = self.reference_image
                kwargs = dict(reference=reference)
            else:
                kwargs = dict(target_spacing=self.target_spacing)
            image_dict[DATA], image_dict[AFFINE] = self.apply_resample(
                *args,
                **kwargs,
            )
        return sample

    @staticmethod
    def apply_resample(
            tensor: torch.Tensor,
            affine: np.ndarray,
            interpolation_order: int,
            target_spacing: Optional[Tuple[float, float, float]] = None,
            reference: Optional[Tuple[torch.Tensor, np.ndarray]] = None,
            ) -> Tuple[torch.Tensor, np.ndarray]:
        array = tensor.numpy()[0]
        if reference is None:
            nii = resample_to_output(
                nib.Nifti1Image(array, affine),
                voxel_sizes=target_spacing,
                order=interpolation_order,
            )
        else:
            reference_tensor, reference_affine = reference
            reference_array = reference_tensor.numpy()[0]
            nii = resample_from_to(
                nib.Nifti1Image(array, affine),
                nib.Nifti1Image(reference_array, reference_affine),
                order=interpolation_order,
            )
        tensor = torch.from_numpy(nii.get_fdata(dtype=np.float32))
        tensor = tensor.unsqueeze(dim=0)
        return tensor, nii.affine
