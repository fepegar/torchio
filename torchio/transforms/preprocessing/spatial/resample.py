from pathlib import Path
from numbers import Number
from typing import Union, Tuple, Optional

import torch
import numpy as np
import SimpleITK as sitk

from ....data.io import sitk_to_nib
from ....data.subject import Subject
from ....typing import TypeTripletFloat
from ....data.image import Image, ScalarImage
from ... import SpatialTransform


TypeSpacing = Union[float, Tuple[float, float, float]]
TypeTarget = Tuple[
    Optional[Union[Image, str]],
    Optional[Tuple[float, float, float]],
]


class Resample(SpatialTransform):
    """Change voxel spacing by resampling.

    Args:
        target: Tuple :math:`(s_h, s_w, s_d)`. If only one value
            :math:`n` is specified, then :math:`s_h = s_w = s_d = n`.
            If a string or :class:`~pathlib.Path` is given,
            all images will be resampled using the image
            with that name as reference or found at the path.
            An instance of :class:`~torchio.Image` can also be passed.
        pre_affine_name: Name of the *image key* (not subject key) storing an
            affine matrix that will be applied to the image header before
            resampling. If ``None``, the image is resampled with an identity
            transform. See usage in the example below.
        image_interpolation: See :ref:`Interpolation`.
        scalars_only: Apply only to instances of :class:`~torchio.ScalarImage`.
            See :class:`~torchio.transforms.RandomAnisotropy`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torch
        >>> import torchio as tio
        >>> transform = tio.Resample(1)                     # resample all images to 1mm iso
        >>> transform = tio.Resample((2, 2, 2))             # resample all images to 2mm iso
        >>> transform = tio.Resample('t1')                  # resample all images to 't1' image space
        >>> # Example: using a precomputed transform to MNI space
        >>> ref_path = tio.datasets.Colin27().t1.path  # this image is in the MNI space, so we can use it as reference/target
        >>> affine_matrix = tio.io.read_matrix('transform_to_mni.txt')  # from a NiftyReg registration. Would also work with e.g. .tfm from SimpleITK
        >>> image = tio.ScalarImage(tensor=torch.rand(1, 256, 256, 180), to_mni=affine_matrix)  # 'to_mni' is an arbitrary name
        >>> transform = tio.Resample(colin.t1.path, pre_affine_name='to_mni')
        >>> transformed = transform(image)  # "image" is now in the MNI space
    """  # noqa: E501
    def __init__(
            self,
            target: Union[TypeSpacing, str, Path, Image, None] = 1,
            image_interpolation: str = 'linear',
            pre_affine_name: Optional[str] = None,
            scalars_only: bool = False,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.target = target
        self.reference_image, self.target_spacing = self.parse_target(target)
        parsed_interpolation = self.parse_interpolation(image_interpolation)
        self.image_interpolation = parsed_interpolation
        self.pre_affine_name = pre_affine_name
        self.scalars_only = scalars_only
        self.args_names = (
            'target',
            'image_interpolation',
            'pre_affine_name',
            'scalars_only',
        )

    def parse_target(self, target: Union[TypeSpacing, str]) -> TypeTarget:
        """
        If target is an existing path, return a torchio.ScalarImage
        If it does not exist, return the string
        If it is not a Path or string or an Image, return None
        """
        if isinstance(target, (str, Path)):
            if Path(target).is_file():
                path = target
                reference_image = ScalarImage(path)
            else:
                reference_image = target
            target_spacing = None
        elif isinstance(target, Image):
            reference_image = target
            target_spacing = reference_image.spacing
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

    @staticmethod
    def check_affine(affine_name: str, image: Image):
        if not isinstance(affine_name, str):
            message = (
                'Affine name argument must be a string,'
                f' not {type(affine_name)}'
            )
            raise TypeError(message)
        if affine_name in image:
            matrix = image[affine_name]
            if not isinstance(matrix, (np.ndarray, torch.Tensor)):
                message = (
                    'The affine matrix must be a NumPy array or PyTorch'
                    f' tensor, not {type(matrix)}'
                )
                raise TypeError(message)
            if matrix.shape != (4, 4):
                message = (
                    'The affine matrix shape must be (4, 4),'
                    f' not {matrix.shape}'
                )
                raise ValueError(message)

    @staticmethod
    def check_affine_key_presence(affine_name: str, subject: Subject):
        for image in subject.get_images(intensity_only=False):
            if affine_name in image:
                return
        message = (
            f'An affine name was given ("{affine_name}"), but it was not found'
            ' in any image in the subject'
        )
        raise ValueError(message)

    def apply_transform(self, subject: Subject) -> Subject:
        use_pre_affine = self.pre_affine_name is not None
        if use_pre_affine:
            self.check_affine_key_presence(self.pre_affine_name, subject)
        for image in self.get_images(subject):
            # Do not resample the reference image if there is one
            if image is self.reference_image:
                continue

            # Choose interpolation
            if not isinstance(image, ScalarImage):
                if self.scalars_only:
                    continue
                interpolation = 'nearest'
            else:
                interpolation = self.image_interpolation
            interpolator = self.get_sitk_interpolator(interpolation)

            # Apply given affine matrix if found in image
            if use_pre_affine and self.pre_affine_name in image:
                self.check_affine(self.pre_affine_name, image)
                matrix = image[self.pre_affine_name]
                if isinstance(matrix, torch.Tensor):
                    matrix = matrix.numpy()
                image.affine = matrix @ image.affine

            floating_itk = image.as_sitk(force_3d=True)

            # Get reference image
            if isinstance(self.reference_image, str):
                try:
                    reference_image = subject[self.reference_image]
                    reference_image_sitk = reference_image.as_sitk()
                except KeyError as error:
                    message = (
                        f'Image name "{self.reference_image}"'
                        f' not found in subject. If "{self.reference_image}"'
                        ' is a path, it does not exist or permission has been'
                        ' denied'
                    )
                    raise ValueError(message) from error
            elif isinstance(self.reference_image, Image):
                reference_image_sitk = self.reference_image.as_sitk(
                    force_3d=True)
            elif self.reference_image is None:  # target is a spacing
                reference_image_sitk = self.get_reference_image(
                    floating_itk,
                    self.target_spacing,
                )
            num_dims_ref = reference_image_sitk.GetDimension()
            num_dims_flo = floating_itk.GetDimension()
            assert num_dims_ref == num_dims_flo

            # Resample
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(interpolator)
            resampler.SetReferenceImage(reference_image_sitk)
            resampled = resampler.Execute(floating_itk)

            array, affine = sitk_to_nib(resampled)
            image.set_data(torch.as_tensor(array))
            image.affine = affine
        return subject

    @staticmethod
    def get_reference_image(
            image: sitk.Image,
            spacing: TypeTripletFloat,
            ) -> sitk.Image:
        old_spacing = np.array(image.GetSpacing())
        new_spacing = np.array(spacing)
        old_size = np.array(image.GetSize())
        new_size = old_size * old_spacing / new_spacing
        new_size = np.ceil(new_size).astype(np.uint16)
        new_size[old_size == 1] = 1  # keep singleton dimensions
        new_origin_index = 0.5 * (new_spacing / old_spacing - 1)
        new_origin_lps = image.TransformContinuousIndexToPhysicalPoint(
            new_origin_index)
        reference = sitk.Image(
            new_size.tolist(),
            image.GetPixelID(),
            image.GetNumberOfComponentsPerPixel(),
        )
        reference.SetDirection(image.GetDirection())
        reference.SetSpacing(new_spacing.tolist())
        reference.SetOrigin(new_origin_lps)
        return reference

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
