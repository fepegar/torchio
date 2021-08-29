from numbers import Real
from typing import Union, Optional, Callable

import torch
from torchio.data.image import ScalarImage
from ....data.subject import Subject
from ...intensity_transform import IntensityTransform


class SlabProjection(IntensityTransform):
    """Project intensities along a given axis, possibly with sliding slabs.

    Args:
        axis: Index for the spatial dimension to project across.
            See :class:`~.torchio.RandomFlip` for information on the accepted
            types.
        slab_thickness: Thickness of slab projections. In other words, the
            number of voxels in the ``axis`` dimension to project across.
            If ``None``, the projection will be done across the entire span of
            the ``axis`` dimension (i.e. ``axis`` dimension will be reduced to
            1).
        stride: Number of voxels to stride along the ``axis`` dimension between
            slab projections. Default is 1.
        projection_type: Type of intensity projection. Possible inputs are
            ``'max'`` (the default), ``'min'``, ``'mean'``, ``'median'``, or
            ``'percentile'``. If ``'percentile'`` is used, the ``percentile``
            argument must also be supplied.
        percentile: Percetile to use for intensity projections. This argument
            is required if ``projection_type`` is ``'percentile'`` and is
            silently ignored otherwise.
        full_slabs_only: Boolean. Should projections be done only for slabs
            that are ``slab_thickness`` thick? Default is ``True``.
            If ``False``, some slabs may not be ``slab_thickness`` thick
            depending on the size of the image, slab thickness, and stride.

    Example:
        >>> import torchio as tio
        >>> ct = tio.datasets.Slicer('CTChest').CT_chest
        >>> axial_mip = tio.SlabProjection("S", slab_thickness=20)
        >>> ct_t = axial_mip(ct)
        >>> ct_t.plot()

    .. plot::

        import torchio as tio
        sub = tio.datasets.Slicer('CTChest')
        ct = sub.CT_chest
        axial_mip = tio.SlabProjection("S", slab_thickness=20)
        ct_mip = axial_mip(ct)
        sub.add_image(ct_mip, 'CT_MIP')
        sub = tio.Clamp(-1000, 1000)(sub)
        sub.plot()

    """
    def __init__(
            self,
            axis: Union[int, str],
            slab_thickness: Optional[int] = None,
            stride: int = 1,
            projection_type: str = 'max',
            percentile: Optional[float] = None,
            full_slabs_only: bool = True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.args_names = (
            'axis', 'slab_thickness', 'stride',
            'projection_type', 'percentile', 'full_slabs_only'
            )
        self.axis = self.parse_axes(axis)[0]
        self.slab_thickness = slab_thickness
        self.stride = stride
        self.projection_fun = self.get_projection_function(projection_type)
        self.projection_type = projection_type
        self.percentile = self.validate_percentile(percentile, projection_type)
        self.full_slabs_only = full_slabs_only

    @staticmethod
    def validate_percentile(percentile, projection_type):
        if not projection_type == 'percentile':
            return percentile
        message = (
            "For projection_type='percentile', `percentile` must be a scalar"
            f' value in the range [0, 1], but "{percentile}" was passed'
        )
        if not isinstance(percentile, Real):
            raise TypeError(message)
        elif 0 <= percentile <= 100:
            return percentile / 100
        else:
            raise ValueError(message)

    @staticmethod
    def get_projection_function(projection_type: str) -> Callable:
        arg_to_function = {
            'max': 'amax',
            'min': 'amin',
            'mean': 'mean',
            'median': 'median',
            'percentile': 'quantile',
        }
        try:
            function_name = arg_to_function[projection_type]
        except KeyError:
            message = (
                f'The projection type must be in {arg_to_function.keys()}, '
                f' but {projection_type} was passed'
            )
            raise ValueError(message)
        projection_function = getattr(torch, function_name)
        return projection_function

    def get_num_slabs(self, axis_span: int) -> int:
        if self.full_slabs_only:
            start_index = 0
            num_slabs = 0
            while start_index + self.slab_thickness <= axis_span:
                num_slabs += 1
                start_index += self.stride
        else:
            num_slabs = torch.ceil(torch.tensor(axis_span) / self.stride)
            num_slabs = int(num_slabs.item())
        return num_slabs

    def apply_transform(self, subject: Subject) -> Subject:
        axis_index = self.ensure_axes_indices(subject, [self.axis])[0]
        for image in self.get_images(subject):
            self.apply_projection(image, axis_index)
        return subject

    def apply_projection(self, image: ScalarImage, axis_index: int) -> None:
        axis_span = image.shape[axis_index]
        if self.slab_thickness is None:
            self.slab_thickness = axis_span
        elif self.slab_thickness > axis_span:
            self.slab_thickness = axis_span
        image.set_data(self.projection(image.data, axis_index, axis_span))

    def projection(
            self,
            tensor: torch.Tensor,
            axis_index: int,
            axis_span: int,
            ) -> torch.Tensor:
        if self.projection_type in ['mean', 'percentile']:
            tensor = tensor.to(torch.float)

        num_slabs = self.get_num_slabs(axis_span)

        slabs = []
        start_index = 0
        end_index = start_index + self.slab_thickness

        for _ in range(num_slabs):
            slab_indices = torch.arange(start_index, end_index)
            slab = tensor.index_select(axis_index, slab_indices)
            if self.projection_type == 'median':
                projected, _ = self.projection_fun(
                    slab, dim=axis_index, keepdim=True)
            elif self.projection_type == 'percentile':
                projected = self.projection_fun(
                    slab, q=self.percentile, dim=axis_index,
                    keepdim=True)
            else:
                projected = self.projection_fun(
                    slab, dim=axis_index, keepdim=True)
            slabs.append(projected)
            start_index += self.stride
            end_index = start_index + self.slab_thickness
            if end_index > axis_span:
                end_index = axis_span

        return torch.cat(slabs, dim=axis_index)
