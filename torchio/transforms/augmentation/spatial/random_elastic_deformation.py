import warnings
from numbers import Number
from typing import Tuple, Optional, Union
import torch
import numpy as np
import SimpleITK as sitk
from ....data.subject import Subject
from ....utils import to_tuple
from ....torchio import INTENSITY, DATA, AFFINE, TYPE
from .. import Interpolation, get_sitk_interpolator
from .. import RandomTransform


SPLINE_ORDER = 3


class RandomElasticDeformation(RandomTransform):
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
            border of the coarse grid will also be set to ``0``.
            If ``2``, displacement of control points at the border of the image
            will also be set to ``0``.
        image_interpolation: See :ref:`Interpolation`.
            Note that this is the interpolation used to compute voxel
            intensities when resampling using the dense displacement field.
            The value of the dense displacement at each voxel is always
            interpolated with cubic B-splines from the values at the control
            points of the coarse grid.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.

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
    example image (they would also be red as we set :py:attr:`locked_borders`
    to ``2``).

    .. warning:: Image folding may occur if the maximum displacement is larger
        than half the coarse grid spacing. The grid spacing can be computed
        using the image bounds in physical space [#]_ and the number of control
        points::

            >>> import numpy as np
            >>> import SimpleITK as sitk
            >>> image = sitk.ReadImage('my_image.nii.gz')
            >>> image.GetSize()
            (512, 512, 139)  # voxels
            >>> image.GetSpacing()
            (0.76, 0.76, 2.50)  # mm
            >>> bounds = np.array(image.GetSize()) * np.array(image.GetSpacing())
            array([390.0, 390.0, 347.5])  # mm
            >>> num_control_points = np.array((7, 7, 6))
            >>> grid_spacing = bounds / (num_control_points - 2)
            >>> grid_spacing
            array([78.0, 78.0, 86.9])  # mm
            >>> potential_folding = grid_spacing / 2
            >>> potential_folding
            array([39.0, 39.0, 43.4])  # mm

        Using a :py:attr:`max_displacement` larger than the computed
        :py:attr:`potential_folding` will raise a :py:class:`RuntimeWarning`.

        .. [#] Technically, :math:`2 \epsilon` should be added to the
            image bounds, where :math:`\epsilon = 2^{-3}` `according to ITK
            source code <https://github.com/InsightSoftwareConsortium/ITK/blob/633f84548311600845d54ab2463d3412194690a8/Modules/Core/Transform/include/itkBSplineTransformInitializer.hxx#L116-L138>`_.
    """

    def __init__(
            self,
            num_control_points: Union[int, Tuple[int, int, int]] = 7,
            max_displacement: Union[float, Tuple[float, float, float]] = 7.5,
            locked_borders: int = 2,
            image_interpolation: str = 'linear',
            p: float = 1,
            seed: Optional[int] = None,
            ):
        super().__init__(p=p, seed=seed)
        self._bspline_transformation = None
        self.num_control_points = to_tuple(num_control_points, length=3)
        self.parse_control_points(self.num_control_points)
        self.max_displacement = to_tuple(max_displacement, length=3)
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
        self.interpolation = self.parse_interpolation(image_interpolation)

    @staticmethod
    def parse_control_points(
            num_control_points: Tuple[int, int, int],
            ) -> None:
        for axis, number in enumerate(num_control_points):
            if not isinstance(number, int) or number < 4:
                message = (
                    f'The number of control points for axis {axis} must be'
                    f' an integer greater than 3, not {number}'
                )
                raise ValueError(message)

    @staticmethod
    def parse_max_displacement(
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

    @staticmethod
    def get_params(
            num_control_points: Tuple[int, int, int],
            max_displacement: Tuple[float, float, float],
            num_locked_borders: int,
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

        return coarse_field.numpy()

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
            warnings.warn(message, RuntimeWarning)

    def apply_transform(self, sample: Subject) -> dict:
        sample.check_consistent_shape()
        bspline_params = self.get_params(
            self.num_control_points,
            self.max_displacement,
            self.num_locked_borders,
        )
        for image in sample.get_images(intensity_only=False):
            if image[TYPE] != INTENSITY:
                interpolation = Interpolation.NEAREST
            else:
                interpolation = self.interpolation
            if image.is_2d():
                bspline_params[..., -3] = 0  # no displacement in LR axis
            image[DATA] = self.apply_bspline_transform(
                image[DATA],
                image[AFFINE],
                bspline_params,
                interpolation,
            )
        random_parameters_dict = {'coarse_grid': bspline_params}
        sample.add_transform(self, random_parameters_dict)
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
