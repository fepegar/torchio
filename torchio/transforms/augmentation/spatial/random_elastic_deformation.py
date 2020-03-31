import warnings
from numbers import Number
from typing import Tuple, Optional, Union
import torch
import numpy as np
import SimpleITK as sitk
from ....utils import is_image_dict, check_consistent_shape, to_tuple
from ....torchio import LABEL, DATA, AFFINE, TYPE
from .. import Interpolation, get_sitk_interpolator
from .. import RandomTransform


SPLINE_ORDER = 3


class RandomElasticDeformation(RandomTransform):
    """Apply dense random elastic deformation.

    A random displacement is assigned to a coarse grid of control points around
    and inside the image. The displacement at each voxel is interpolated from
    the coarse grid using cubic B-splines.

    The `'Deformable Registration' <https://www.sciencedirect.com/topics/computer-science/deformable-registration>`_
    topic on ScienceDirect contains useful articles explaining interpolation of
    displacement fields using cubic B-splines.

    Args:
        num_control_points: Number of control points along each dimension of
            the coarse grid :math:`(n_x, n_y, n_z)`.
            If a single value :math:`n` is passed,
            then :math:`n_x = n_y = n_z = n`.
            Smaller numbers generate smoother deformations.
            The minimum number of control points is ``4`` as this transform
            uses cubic B-splines to interpolate displacement.
        max_displacement: Maximum displacement along each dimension at each
            control point :math:`(D_x, D_y, D_z)`.
            The displacement along dimension :math:`i` at each control point is
            :math:`d_i \sim \mathcal{U}(0, D_i)`.
            If a single value :math:`D` is passed,
            then :math:`D_x = D_y = D_z = D`.
            Note that the total maximum displacement would actually be
            :math:`D_{max} = \sqrt{D_x^2 + D_y^2 + D_z^2}`.
        locked_borders: If ``0``, all displacement vectors are kept.
            If ``1``, displacement of control points at the
            border of the coarse grid will also be set to ``0``.
            If ``2``, displacement of control points at the border of the image
            will also be set to ``0``.
        image_interpolation: Value in
            :py:class:`torchio.transforms.interpolation.Interpolation`.
            Note that this is the interpolation used to compute voxel
            intensities when resampling using the displacement field.
            The displacement at each voxel is always interpolated with cubic
            B-splines from the coarse grid of control points.
        proportion_to_augment: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.

    `This gist <https://gist.github.com/fepegar/b723d15de620cd2a3a4dbd71e491b59d>`_
    can also be used to better understand the meaning of the parameters.

    This is an example from the
    `3D Slicer registration FAQ <https://www.slicer.org/wiki/Documentation/4.10/FAQ/Registration#What.27s_the_BSpline_Grid_Size.3F>`_.

    .. image:: https://www.slicer.org/w/img_auth.php/6/6f/RegLib_BSplineGridModel.png
        :alt: B-spline example from 3D Slicer documentation

    To obtain a generate a similar grid of control points with TorchIO,
    the transform can be instantiated as follows::

        >>> from torchio import RandomElasticDeformation
        >>> transform = RandomElasticDeformation(
        ...     num_control_points=(7, 7, 7),  # or just 7
        ...     locked_borders=2,
        ... )

    Note that control points outside the image bounds are not showed in the
    example image (they would also be red as we set :py:attr:`locked_borders`
    to 2).
    """

    def __init__(
            self,
            num_control_points: Union[int, Tuple[int, int, int]] = 7,
            max_displacement: Union[float, Tuple[float, float, float]] = 7.5,
            locked_borders: int = 2,
            image_interpolation: Interpolation = Interpolation.LINEAR,
            proportion_to_augment: float = 1,
            seed: Optional[int] = None,
            deformation_std: Union[None, float, Tuple[float, float, float]] = None,
            ):
        super().__init__(seed=seed)
        self._bspline_transformation = None
        self.num_control_points = to_tuple(num_control_points, n=3)
        self.parse_control_points(self.num_control_points)
        self.max_displacement = to_tuple(max_displacement, n=3)
        self.parse_max_displacement(self.max_displacement)
        self.num_locked_borders = locked_borders
        if locked_borders not in (0, 1, 2):
            raise ValueError('locked_borders must be 0, 1, or 2')
        if locked_borders == 2 and 4 in self.num_control_points:
            message = (
                'Setting locked_borders to 2 and using less than 5 control'
                'points results in an identity transform. Lock fewer borders'
                ' or use more control points.'
            )
            raise ValueError(message)
        self.proportion_to_augment = self.parse_probability(
            proportion_to_augment,
            'proportion_to_augment',
        )
        self.interpolation = self.parse_interpolation(image_interpolation)
        if deformation_std is not None:
            message = (
                'The argument "deformation_std" is deprecated.'
                ' Use "max_displacement" instead'
            )
            warnings.warn(message, DeprecationWarning)
            self.max_displacement = deformation_std

    def parse_control_points(
            self,
            num_control_points: Tuple[int, int, int],
            ) -> None:
        for axis, n in enumerate(num_control_points):
            if not isinstance(n, int) or n < 4:
                message = (
                    f'The number of control points for axis {axis} must be'
                    f' an integer larger than 3, not {n}'
                )
                raise ValueError(message)

    def parse_max_displacement(
            self,
            max_displacement: Tuple[float, float, float],
            ) -> None:
        for axis, n in enumerate(max_displacement):
            if not isinstance(n, Number) or n < 0:
                message = (
                    'The maximum displacement at each control point'
                    f' for axis {axis} must be'
                    f' a number greater or equal to 0, not {n}'
                )
                raise ValueError(message)

    @staticmethod
    def get_params(
            num_control_points: Tuple[int, int, int],
            max_displacement: Tuple[float, float, float],
            num_locked_borders: int,
            probability: float,
            ) -> Tuple:
        grid_shape = num_control_points
        num_dimensions = 3
        coarse_field = torch.rand(*grid_shape, num_dimensions)  # [0, 1)
        coarse_field -= 0.5  # [-0.5, 0.5)
        coarse_field *= 2  # [-1, 1]
        for dimension in range(3):
            # [-max_displacement, max_displacement)
            coarse_field[..., dimension] *= max_displacement[dimension]

        # Set displacement to 0 at the borders
        for i in range(num_locked_borders):
            coarse_field[i, :] = 0
            coarse_field[-1 - i, :] = 0
            coarse_field[:, i] = 0
            coarse_field[:, -1 - i] = 0

        do_augmentation = torch.rand(1) < probability
        return do_augmentation, coarse_field.numpy()

    @staticmethod
    def get_bspline_transform(
            image: sitk.Image,
            num_control_points: Tuple[int, int, int],
            coarse_field: np.ndarray,
            ) -> sitk.BSplineTransformInitializer:
        mesh_shape = [n - SPLINE_ORDER for n in num_control_points]
        bspline_transform = sitk.BSplineTransformInitializer(image, mesh_shape)
        parameters = coarse_field.flatten(order='F').tolist()
        bspline_transform.SetParameters(parameters)
        return bspline_transform

    @staticmethod
    def parse_free_form_transform(transform, max_displacement):
        """Issue a warning is possible folding is detected."""
        coefficient_images = transform.GetCoefficientImages()
        grid_spacing = coefficient_images[0].GetSpacing()
        conflicts = np.array(max_displacement) > np.array(grid_spacing) / 2
        if np.any(conflicts):
            where, = np.where(conflicts)
            message = (
                'The maximum displacement is larger than the coarse grid'
                f' spacing for dimensions: {where.tolist()}, so folding may'
                ' occur. Choose fewer control points or a smaller'
                ' maximum displacement'
            )
            warnings.warn(message)

    def apply_transform(self, sample: dict) -> dict:
        check_consistent_shape(sample)
        bspline_params = None
        sample['random_elastic_deformation'] = {}
        params_dict = sample['random_elastic_deformation']

        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            if image_dict[TYPE] == LABEL:
                interpolation = Interpolation.NEAREST
            else:
                interpolation = self.interpolation
            if bspline_params is None:
                image = self.nib_to_sitk(
                    image_dict[DATA][0],
                    image_dict[AFFINE],
                )
                do_augmentation, bspline_params = self.get_params(
                    self.num_control_points,
                    self.max_displacement,
                    self.num_locked_borders,
                    self.proportion_to_augment,
                )
                params_dict['bspline_params'] = bspline_params
                params_dict['do_augmentation'] = int(do_augmentation)
                if not do_augmentation:
                    return sample
            image_dict[DATA] = self.apply_bspline_transform(
                image_dict[DATA],
                image_dict[AFFINE],
                bspline_params,
                interpolation,
            )
        return sample

    def apply_bspline_transform(
            self,
            tensor: torch.Tensor,
            affine: np.ndarray,
            bspline_params: np.ndarray,
            interpolation: Interpolation,
            ) -> torch.Tensor:
        assert tensor.ndim == 4
        assert len(tensor) == 1
        image = self.nib_to_sitk(tensor[0], affine)
        floating = reference = image
        bspline_transform = self.get_bspline_transform(
            image,
            self.num_control_points,
            bspline_params,
        )
        self.parse_free_form_transform(
            bspline_transform, self.max_displacement)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetTransform(bspline_transform)
        resampler.SetInterpolator(get_sitk_interpolator(interpolation))
        resampler.SetDefaultPixelValue(tensor.min().item())
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampled = resampler.Execute(floating)

        np_array = sitk.GetArrayFromImage(resampled)
        np_array = np_array.transpose()  # ITK to NumPy
        tensor[0] = torch.from_numpy(np_array)
        return tensor
