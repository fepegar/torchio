from pathlib import Path
from numbers import Number
from typing import Union, Tuple, Optional
from collections.abc import Iterable

import torch
import numpy as np
import SimpleITK as sitk

from ....data.io import sitk_to_nib, get_sitk_metadata_from_ras_affine
from ....data.subject import Subject
from ....typing import TypeTripletFloat, TypePath
from ....data.image import Image, ScalarImage
from ... import SpatialTransform


TypeSpacing = Union[float, Tuple[float, float, float]]


class Resample(SpatialTransform):
    """Resample image to a different physical space.

    This is a powerful transform that can be used to change the image shape
    or spatial metadata, or to apply a spatial transformation.

    Args:
        target: Argument to define the output space. Can be one of:

            - Output spacing :math:`(s_w, s_h, s_d)`, in mm. If only one value
              :math:`s` is specified, then :math:`s_w = s_h = s_d = s`.

            - Path to an image that will be used as reference.

            - Instance of :class:`~torchio.Image`.

            - Name of an image key in the subject.

            - Tuple ``(spatial_shape, affine)`` defining the output space.

        pre_affine_name: Name of the *image key* (not subject key) storing an
            affine matrix that will be applied to the image header before
            resampling. If ``None``, the image is resampled with an identity
            transform. See usage in the example below.
        image_interpolation: See :ref:`Interpolation`.
        scalars_only: Apply only to instances of :class:`~torchio.ScalarImage`.
            Used internally by :class:`~torchio.transforms.RandomAnisotropy`.
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
        >>> transform = tio.Resample(colin.t1.path, pre_affine_name='to_mni')  # nearest neighbor interpolation is used for label maps
        >>> transformed = transform(image)  # "image" is now in the MNI space

    .. plot::

        import torchio as tio
        subject = tio.datasets.FPG()
        subject.remove_image('seg')
        resample = tio.Resample(8)
        t1_resampled = resample(subject.t1)
        subject.add_image(t1_resampled, 'Downsampled')
        subject.plot()

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
        self.image_interpolation = self.parse_interpolation(
            image_interpolation)
        self.pre_affine_name = pre_affine_name
        self.scalars_only = scalars_only
        self.args_names = (
            'target',
            'image_interpolation',
            'pre_affine_name',
            'scalars_only',
        )

    @staticmethod
    def _parse_spacing(spacing: TypeSpacing) -> Tuple[float, float, float]:
        if isinstance(spacing, Iterable) and len(spacing) == 3:
            result = spacing
        elif isinstance(spacing, Number):
            result = 3 * (spacing,)
        else:
            message = (
                'Target must be a string, a positive number'
                f' or a sequence of positive numbers, not {type(spacing)}'
            )
            raise ValueError(message)
        if np.any(np.array(spacing) <= 0):
            message = f'Spacing must be strictly positive, not "{spacing}"'
            raise ValueError(message)
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
            # Do not resample the reference image if it is in the subject
            if self.target is image:
                continue
            try:
                target_image = subject[self.target]
                if target_image is image:
                    continue
            except (KeyError, TypeError):
                pass

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

            floating_sitk = image.as_sitk(force_3d=True)

            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(interpolator)
            self._set_resampler_reference(
                resampler,
                self.target,
                floating_sitk,
                subject,
            )
            resampled = resampler.Execute(floating_sitk)

            array, affine = sitk_to_nib(resampled)
            image.set_data(torch.as_tensor(array))
            image.affine = affine
        return subject

    def _set_resampler_reference(
            self,
            resampler: sitk.ResampleImageFilter,
            target: Union[TypeSpacing, TypePath, Image],
            floating_sitk,
            subject,
            ):
        # Target can be:
        # 1) An instance of torchio.Image
        # 2) An instance of pathlib.Path
        # 3) A string, which could be a path or an image in subject
        # 3) A string, which could be a path or an image in subject
        # 4) A number or sequence of numbers for spacing
        # 5) A tuple of shape, affine
        # The fourth case is the different one
        if isinstance(target, (str, Path, Image)):
            if isinstance(target, Image):
                # It's a TorchIO image
                image = target
            elif Path(target).is_file():
                # It's an existing file
                path = target
                image = ScalarImage(path)
            else:  # assume it's the name of an image in the subject
                try:
                    image = subject[target]
                except KeyError as error:
                    message = (
                        f'Image name "{target}" not found in subject.'
                        f' If "{target}" is a path, it does not exist or'
                        ' permission has been denied'
                    )
                    raise ValueError(message) from error
            self._set_resampler_from_shape_affine(
                resampler,
                image.spatial_shape,
                image.affine,
            )
        elif isinstance(target, Number):  # one number for target was passed
            self._set_resampler_from_spacing(resampler, target, floating_sitk)
        elif isinstance(target, Iterable) and len(target) == 2:
            shape, affine = target
            if not (isinstance(shape, Iterable) and len(shape) == 3):
                message = (
                    f'Target shape must be a sequence of three integers, but'
                    f' "{shape}" was passed'
                )
                raise RuntimeError(message)
            if not affine.shape == (4, 4):
                message = (
                    f'Target affine must have shape (4, 4) but the following'
                    f' was passed:\n{shape}'
                )
                raise RuntimeError(message)
            self._set_resampler_from_shape_affine(
                resampler,
                shape,
                affine,
            )
        elif isinstance(target, Iterable) and len(target) == 3:
            self._set_resampler_from_spacing(resampler, target, floating_sitk)
        else:
            raise RuntimeError(f'Target not understood: "{target}"')

    def _set_resampler_from_shape_affine(self, resampler, shape, affine):
        origin, spacing, direction = get_sitk_metadata_from_ras_affine(affine)
        resampler.SetOutputDirection(direction)
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputSpacing(spacing)
        resampler.SetSize(shape)

    def _set_resampler_from_spacing(self, resampler, target, floating_sitk):
        target_spacing = self._parse_spacing(target)
        reference_image = self.get_reference_image(
            floating_sitk,
            target_spacing,
        )
        resampler.SetReferenceImage(reference_image)

    @staticmethod
    def get_reference_image(
            floating_sitk: sitk.Image,
            spacing: TypeTripletFloat,
            ) -> sitk.Image:
        old_spacing = np.array(floating_sitk.GetSpacing())
        new_spacing = np.array(spacing)
        old_size = np.array(floating_sitk.GetSize())
        new_size = old_size * old_spacing / new_spacing
        new_size = np.ceil(new_size).astype(np.uint16)
        new_size[old_size == 1] = 1  # keep singleton dimensions
        new_origin_index = 0.5 * (new_spacing / old_spacing - 1)
        new_origin_lps = floating_sitk.TransformContinuousIndexToPhysicalPoint(
            new_origin_index)
        reference = sitk.Image(
            new_size.tolist(),
            floating_sitk.GetPixelID(),
            floating_sitk.GetNumberOfComponentsPerPixel(),
        )
        reference.SetDirection(floating_sitk.GetDirection())
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
