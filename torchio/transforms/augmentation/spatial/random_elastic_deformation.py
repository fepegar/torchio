import warnings
from numbers import Number
from typing import Tuple, Union, Sequence

import torch
import numpy as np
import SimpleITK as sitk

from ....utils import to_tuple
from ....data.io import nib_to_sitk
from ....data.subject import Subject
from ....data.image import ScalarImage
from ....typing import TypeTripletInt, TypeTripletFloat
from ... import SpatialTransform
from .. import RandomTransform


SPLINE_ORDER = 3


class RandomElasticDeformation(RandomTransform, SpatialTransform):
    r"""Apply dense random elastic deformation.

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
            border of the coarse grid will be set to ``0``.
            If ``2``, displacement of control points at the border of the image
            (red dots in the image below) will also be set to ``0``.
        image_interpolation: See :ref:`Interpolation`.
            Note that this is the interpolation used to compute voxel
            intensities when resampling using the dense displacement field.
            The value of the dense displacement at each voxel is always
            interpolated with cubic B-splines from the values at the control
            points of the coarse grid.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    `This gist <https://gist.github.com/fepegar/b723d15de620cd2a3a4dbd71e491b59d>`_
    can also be used to better understand the meaning of the parameters.

    This is an example from the
    `3D Slicer registration FAQ <https://www.slicer.org/wiki/Documentation/4.10/FAQ/Registration#What.27s_the_BSpline_Grid_Size.3F>`_.

    .. image:: https://www.slicer.org/w/img_auth.php/6/6f/RegLib_BSplineGridModel.png
        :alt: B-spline example from 3D Slicer documentation

    To generate a similar grid of control points with TorchIO,
    the transform can be instantiated as follows::

        >>> from torchio import RandomElasticDeformation
        >>> transform = RandomElasticDeformation(
        ...     num_control_points=(7, 7, 7),  # or just 7
        ...     locked_borders=2,
        ... )

    Note that control points outside the image bounds are not showed in the
    example image (they would also be red as we set :attr:`locked_borders`
    to ``2``).

    .. warning:: Image folding may occur if the maximum displacement is larger
        than half the coarse grid spacing. The grid spacing can be computed
        using the image bounds in physical space [#]_ and the number of control
        points::

            >>> import numpy as np
            >>> import torchio as tio
            >>> image = tio.datasets.Slicer().MRHead.as_sitk()
            >>> image.GetSize()  # in voxels
            (256, 256, 130)
            >>> image.GetSpacing()  # in mm
            (1.0, 1.0, 1.2999954223632812)
            >>> bounds = np.array(image.GetSize()) * np.array(image.GetSpacing())
            >>> bounds  # mm
            array([256.        , 256.        , 168.99940491])
            >>> num_control_points = np.array((7, 7, 6))
            >>> grid_spacing = bounds / (num_control_points - 2)
            >>> grid_spacing
            array([51.2       , 51.2       , 42.24985123])
            >>> potential_folding = grid_spacing / 2
            >>> potential_folding  # mm
            array([25.6       , 25.6       , 21.12492561])

        Using a :attr:`max_displacement` larger than the computed
        :attr:`potential_folding` will raise a :class:`RuntimeWarning`.

        .. [#] Technically, :math:`2 \epsilon` should be added to the
            image bounds, where :math:`\epsilon = 2^{-3}` `according to ITK
            source code <https://github.com/InsightSoftwareConsortium/ITK/blob/633f84548311600845d54ab2463d3412194690a8/Modules/Core/Transform/include/itkBSplineTransformInitializer.hxx#L116-L138>`_.
    """  # noqa: E501

    def __init__(
            self,
            num_control_points: Union[int, Tuple[int, int, int]] = 7,
            max_displacement: Union[float, Tuple[float, float, float]] = 7.5,
            locked_borders: int = 2,
            image_interpolation: str = 'linear',
            **kwargs
            ):
        super().__init__(**kwargs)
        self._bspline_transformation = None
        self.num_control_points = to_tuple(num_control_points, length=3)
        _parse_num_control_points(self.num_control_points)
        self.max_displacement = to_tuple(max_displacement, length=3)
        _parse_max_displacement(self.max_displacement)
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
        self.image_interpolation = self.parse_interpolation(
            image_interpolation)

    @staticmethod
    def get_params(
            num_control_points: TypeTripletInt,
            max_displacement: Tuple[float, float, float],
            num_locked_borders: int,
            ) -> np.ndarray:
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

        return coarse_field.numpy()

    def apply_transform(self, subject: Subject) -> Subject:
        subject.check_consistent_spatial_shape()
        control_points = self.get_params(
            self.num_control_points,
            self.max_displacement,
            self.num_locked_borders,
        )

        arguments = {
            'control_points': control_points,
            'max_displacement': self.max_displacement,
            'image_interpolation': self.image_interpolation,
        }

        transform = ElasticDeformation(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed


class ElasticDeformation(SpatialTransform):
    r"""Apply dense elastic deformation.

    Args:
        control_points:
        max_displacement:
        image_interpolation: See :ref:`Interpolation`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(
            self,
            control_points: np.ndarray,
            max_displacement: TypeTripletFloat,
            image_interpolation: str = 'linear',
            **kwargs
            ):
        super().__init__(**kwargs)
        self.control_points = control_points
        self.max_displacement = max_displacement
        self.image_interpolation = self.parse_interpolation(
            image_interpolation)
        self.invert_transform = False
        self.args_names = (
            'control_points',
            'image_interpolation',
            'max_displacement',
        )

    @staticmethod
    def get_bspline_transform(
            image: sitk.Image,
            control_points: np.ndarray,
            ) -> sitk.BSplineTransformInitializer:
        num_control_points = control_points.shape[:-1]
        mesh_shape = [n - SPLINE_ORDER for n in num_control_points]
        bspline_transform = sitk.BSplineTransformInitializer(image, mesh_shape)
        parameters = control_points.flatten(order='F').tolist()
        bspline_transform.SetParameters(parameters)
        return bspline_transform

    @staticmethod
    def parse_free_form_transform(
            transform: sitk.Transform,
            max_displacement: Sequence[TypeTripletInt],
            ) -> None:
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
            warnings.warn(message, RuntimeWarning)

    def apply_transform(self, subject: Subject) -> Subject:
        no_displacement = not any(self.max_displacement)
        if no_displacement:
            return subject
        subject.check_consistent_spatial_shape()
        control_points = self.control_points.copy()
        if self.invert_transform:
            control_points *= -1
        for image in self.get_images(subject):
            if not isinstance(image, ScalarImage):
                interpolation = 'nearest'
            else:
                interpolation = self.image_interpolation
            if image.is_2d():
                control_points[..., -1] = 0  # no displacement in IS axis
            transformed = self.apply_bspline_transform(
                image.data,
                image.affine,
                control_points,
                interpolation,
            )
            image.set_data(transformed)
        return subject

    def apply_bspline_transform(
            self,
            tensor: torch.Tensor,
            affine: np.ndarray,
            control_points: np.ndarray,
            interpolation: str,
            ) -> torch.Tensor:
        assert tensor.dim() == 4
        results = []
        for component in tensor:
            image = nib_to_sitk(component[np.newaxis], affine, force_3d=True)
            floating = reference = image
            bspline_transform = self.get_bspline_transform(
                image,
                control_points,
            )
            self.parse_free_form_transform(
                bspline_transform,
                self.max_displacement,
            )
            interpolator = self.get_sitk_interpolator(interpolation)
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(reference)
            resampler.SetTransform(bspline_transform)
            resampler.SetInterpolator(interpolator)
            resampler.SetDefaultPixelValue(component.min().item())
            resampler.SetOutputPixelType(sitk.sitkFloat32)
            resampled = resampler.Execute(floating)
            result, _ = self.sitk_to_nib(resampled)
            results.append(torch.from_numpy(result))
        tensor = torch.cat(results)
        return tensor


def _parse_num_control_points(
        num_control_points: TypeTripletInt,
        ) -> None:
    for axis, number in enumerate(num_control_points):
        if not isinstance(number, int) or number < 4:
            message = (
                f'The number of control points for axis {axis} must be'
                f' an integer greater than 3, not {number}'
            )
            raise ValueError(message)


def _parse_max_displacement(
        max_displacement: Tuple[float, float, float],
        ) -> None:
    for axis, number in enumerate(max_displacement):
        if not isinstance(number, Number) or number < 0:
            message = (
                'The maximum displacement at each control point'
                f' for axis {axis} must be'
                f' a number greater or equal to 0, not {number}'
            )
            raise ValueError(message)
