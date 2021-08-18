import torch
from torchio.data.image import ScalarImage
from ....data.subject import Subject
from ...intensity_transform import IntensityTransform
from typing import Optional
from math import ceil


class Projection(IntensityTransform):
    """Project intensities along a given axis, possibly with sliding slabs.

    Args:
        axis: Possible inputs are ``'Left'``, ``'Right'``, ``'Anterior'``,
                ``'Posterior'``, ``'Inferior'``, ``'Superior'``. Lower-case
                versions and first letters are also valid, as only the first
                letter will be used.
        slab_thickness: Thickness of slab projections. In other words, the
            number of voxels in the ``axis`` dimension to project across.
            If ``None``, the projection will be done across the entire span of
            the ``axis`` dimension (i.e. ``axis`` dimension will be reduced to
            1).
        stride: Number of voxels to stride along the ``axis`` dimension between
            slab projections.
        projection_type: Type of intensity projection. Possible inputs are
            ``'max'`` (the default), ``'min'``, ``'mean'``, ``'median'``, or
            ``'quantile'``. If ``'quantile'`` is used, ``q`` must also be
            supplied.
        q: Quantile to use for intensity projections. This argument is required
            if ``projection_type`` is ``'quantile'`` and is silently ignored
            otherwise.
        full_slabs_only: Boolean. Should projections be done only for slabs
            that are ``slab_thickness`` thick? Default is ``True``.
            If ``False``, some slabs may not be ``slab_thickness`` thick
            depending on the size of the image, slab thickness, and stride.

    Example:
        >>> import torchio as tio
        >>> sub = tio.datasets.Colin27()
        >>> axial_mips = tio.Projection("S", slab_thickness=20)
        >>> sub_t = axial_mips(sub)
        >>> sub_t.t1.plot()
    """
    def __init__(
            self,
            axis: str,
            slab_thickness: Optional[int] = None,
            stride: Optional[int] = 1,
            projection_type: Optional[str] = 'max',
            q: Optional[float] = None,
            full_slabs_only: Optional[bool] = True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.args_names = (
            'axis', 'slab_thickness', 'stride',
            'projection_type', 'q', 'full_slabs_only'
            )
        self.axis = axis
        self.slab_thickness = slab_thickness
        self.stride = stride
        self.projection_type = projection_type
        self.q = q
        self.full_slabs_only = full_slabs_only
        self.projection_fun = self.get_projection_function()

    def get_projection_function(self):
        if self.projection_type == 'max':
            projection_fun = torch.max
        elif self.projection_type == 'min':
            projection_fun = torch.min
        elif self.projection_type == 'mean':
            projection_fun = torch.mean
        elif self.projection_type == 'median':
            projection_fun = torch.median
        elif self.projection_type == 'quantile':
            projection_fun = torch.quantile
            self.validate_quantile()
        else:
            message = (
                '`projection_type` must be one of "max", "min", "mean",'
                ' "median", or "quantile".'
                )
            raise ValueError(message)
        return projection_fun

    def validate_quantile(self):
        message = (
            'For `projection_type="quantile"`, `q` must be a scalar value'
            f'between 0 and 1, not {self.q}.'
            )
        if self.q is None:
            raise ValueError(message)
        elif 0 < self.q < 1:
            pass
        else:
            raise ValueError(message)

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            self.apply_projection(image)
        return subject

    def apply_projection(self, image: ScalarImage) -> None:
        self.axis_index = image.axis_name_to_index(self.axis)
        self.axis_span = image.shape[self.axis_index]
        if self.slab_thickness is None:
            self.slab_thickness = self.axis_span
        elif self.slab_thickness > self.axis_span:
            self.slab_thickness = self.axis_span
        image.set_data(self.projection(image.data))

    def projection(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.full_slabs_only:
            start_index = 0
            num_slabs = 0
            while start_index + self.slab_thickness <= self.axis_span:
                num_slabs += 1
                start_index += self.stride
        else:
            num_slabs = ceil(self.axis_span / self.stride)

        slabs = []
        start_index = 0
        end_index = start_index + self.slab_thickness

        for _ in range(num_slabs):
            slab_indices = torch.tensor(list(range(start_index, end_index)))
            slab = tensor.index_select(self.axis_index, slab_indices)
            if self.projection_type == 'mean':
                projected = self.projection_fun(
                    slab, dim=self.axis_index, keepdim=True)
            elif self.projection_type == 'quantile':
                projected = self.projection_fun(
                    slab, q=self.q, dim=self.axis_index, keepdim=True)
            else:
                projected, _ = self.projection_fun(
                    slab, dim=self.axis_index, keepdim=True)
            slabs.append(projected)
            start_index += self.stride
            end_index = start_index + self.slab_thickness
            if end_index > self.axis_span:
                end_index = self.axis_span

        return torch.cat(slabs, dim=self.axis_index)
