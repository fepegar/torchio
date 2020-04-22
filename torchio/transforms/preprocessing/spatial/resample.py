from numbers import Number
from typing import Union, Tuple, Optional
import torch
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
from ....data.subject import Subject
from ....utils import is_image_dict
from ....torchio import LABEL, DATA, AFFINE, TYPE
from ... import Interpolation
from ... import Transform


TypeSpacing = Union[float, Tuple[float, float, float]]


class Resample(Transform):
    """Change voxel spacing by resampling.

    Args:
        target: Tuple :math:`(s_d, s_h, s_w)`. If only one value
            :math:`n` is specified, then :math:`s_d = s_h = s_w = n`.
            If a string is given, all images will be resampled using the image
            with that name as reference.
        antialiasing: (Not implemented yet).
        image_interpolation: Member of :py:class:`torchio.Interpolation`.
            Supported interpolation techniques for resampling are
            :py:attr:`torchio.Interpolation.NEAREST`,
            :py:attr:`torchio.Interpolation.LINEAR` and
            :py:attr:`torchio.Interpolation.BSPLINE`.
        p: Probability that this transform will be applied.

    .. note:: Resampling is performed using
        :py:meth:`nibabel.processing.resample_to_output` or
        :py:meth:`nibabel.processing.resample_from_to`, depending on whether
        the target is a spacing or a reference image.

    Example:
        >>> from torchio.transforms import Resample
        >>> transform = Resample(1)          # resample all images to 1mm iso
        >>> transform = Resample((1, 1, 1))  # resample all images to 1mm iso
        >>> transform = Resample('t1')       # resample all images to 't1' image space

    """
    def __init__(
            self,
            target: Union[TypeSpacing, str],
            antialiasing: bool = True,
            image_interpolation: Interpolation = Interpolation.LINEAR,
            p: float = 1,
            ):
        super().__init__(p=p)
        self.target_spacing: Tuple[float, float, float]
        self.reference_image: str
        self.parse_target(target)
        self.antialiasing = antialiasing
        self.interpolation_order = self.parse_interpolation(
            image_interpolation)

    def parse_target(self, target: Union[TypeSpacing, str]):
        if isinstance(target, str):
            self.reference_image = target
            self.target_spacing = None
        else:
            self.reference_image = None
            self.target_spacing = self.parse_spacing(target)

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

    @staticmethod
    def parse_interpolation(interpolation: Interpolation) -> int:
        if interpolation == Interpolation.NEAREST:
            order = 0
        elif interpolation == Interpolation.LINEAR:
            order = 1
        elif interpolation == Interpolation.BSPLINE:
            order = 3
        else:
            message = f'Interpolation not implemented yet: {interpolation}'
            raise NotImplementedError(message)
        return order

    @staticmethod
    def check_reference_image(reference_image: str, sample: Subject):
        if not isinstance(reference_image, str):
            message = f'reference_image argument should be of type str, type {type(reference_image)} was given'
            raise TypeError(message)
        if reference_image not in sample.keys():
            message = f'reference_image=\'{reference_image}\' not present in sample, only these keys were found: {sample.keys()}'
            raise ValueError(message)

    def apply_transform(self, sample: Subject) -> dict:
        use_reference = self.reference_image is not None
        for image_name, image_dict in sample.items():
            # Do not resample the reference image if there is one
            if use_reference and image_name == self.reference_image:
                continue
            if not is_image_dict(image_dict):
                continue

            # Choose interpolator
            if image_dict[TYPE] == LABEL:
                interpolation_order = 0  # nearest neighbor
            else:
                interpolation_order = self.interpolation_order

            # Resample
            args = image_dict[DATA], image_dict[AFFINE], interpolation_order
            if use_reference:
                try:
                    ref_image_dict = sample[self.reference_image]
                except KeyError as error:
                    message = (
                        f'Reference name "{self.reference_image}"'
                        ' not found in sample'
                    )
                    raise ValueError(message) from error
                reference = ref_image_dict[DATA], ref_image_dict[AFFINE]
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
