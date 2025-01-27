from collections import defaultdict
from collections.abc import Iterable
from typing import Optional
from typing import Union

import numpy as np
import torch

from ....data.subject import Subject
from ...fourier import FourierTransform
from ...intensity_transform import IntensityTransform
from .. import RandomTransform


class RandomGhosting(RandomTransform, IntensityTransform):
    r"""Add random MRI ghosting artifact.

    Discrete "ghost" artifacts may occur along the phase-encode direction
    whenever the position or signal intensity of imaged structures within the
    field-of-view vary or move in a regular (periodic) fashion. Pulsatile flow
    of blood or CSF, cardiac motion, and respiratory motion are the most
    important patient-related causes of ghost artifacts in clinical MR imaging
    (from `mriquestions.com`_).

    .. _mriquestions.com: https://mriquestions.com/why-discrete-ghosts.html

    Args:
        num_ghosts: Number of 'ghosts' :math:`n` in the image.
            If :attr:`num_ghosts` is a tuple :math:`(a, b)`, then
            :math:`n \sim \mathcal{U}(a, b) \cap \mathbb{N}`.
            If only one value :math:`d` is provided,
            :math:`n \sim \mathcal{U}(0, d) \cap \mathbb{N}`.
        axes: Axis along which the ghosts will be created. If
            :attr:`axes` is a tuple, the axis will be randomly chosen
            from the passed values. Anatomical labels may also be used (see
            :class:`~torchio.transforms.augmentation.RandomFlip`).
        intensity: Positive number representing the artifact strength
            :math:`s` with respect to the maximum of the :math:`k`-space.
            If ``0``, the ghosts will not be visible. If a tuple
            :math:`(a, b)` is provided then :math:`s \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`s \sim \mathcal{U}(0, d)`.
        restore: Number between ``0`` and ``1`` indicating how much of the
            :math:`k`-space center should be restored after removing the planes
            that generate the artifact. If ``None``, only the central slice
            will be restored. If a tuple :math:`(a, b)` is provided then
            :math:`r \sim \mathcal{U}(a, b)`. If only one value :math:`d` is
            provided, :math:`r \sim \mathcal{U}(0, d)`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. note:: The execution time of this transform does not depend on the
        number of ghosts.
    """

    def __init__(
        self,
        num_ghosts: Union[int, tuple[int, int]] = (4, 10),
        axes: Union[int, tuple[int, ...]] = (0, 1, 2),
        intensity: Union[float, tuple[float, float]] = (0.5, 1),
        restore: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(axes, tuple):
            try:
                axes = tuple(axes)  # type: ignore[arg-type]
            except TypeError:
                axes = (axes,)  # type: ignore[assignment]
        assert isinstance(axes, Iterable)
        for axis in axes:
            if not isinstance(axis, str) and axis not in (0, 1, 2):
                raise ValueError(f'Axes must be in (0, 1, 2), not "{axes}"')
        self.axes = axes
        self.num_ghosts_range = self._parse_range(
            num_ghosts,
            'num_ghosts',
            min_constraint=0,
            type_constraint=int,
        )
        self.intensity_range = self._parse_range(
            intensity,
            'intensity_range',
            min_constraint=0,
        )
        if restore is None:
            self.restore = None
        else:
            self.restore = self._parse_range(
                restore,
                'restore',
                min_constraint=0,
                max_constraint=1,
            )

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict = self.get_images_dict(subject)
        if not images_dict:
            return subject

        if any(isinstance(axis, str) for axis in self.axes):
            subject.check_consistent_orientation()

        arguments: dict[str, dict] = defaultdict(dict)
        for name, image in images_dict.items():
            is_2d = image.is_2d()
            axes = [a for a in self.axes if a != 2] if is_2d else self.axes
            min_ghosts, max_ghosts = self.num_ghosts_range
            params = self.get_params(
                (int(min_ghosts), int(max_ghosts)),
                axes,  # type: ignore[arg-type]
                self.intensity_range,
                self.restore,
            )
            num_ghosts_param, axis_param, intensity_param, restore_param = params
            arguments['num_ghosts'][name] = num_ghosts_param
            arguments['axis'][name] = axis_param
            arguments['intensity'][name] = intensity_param
            arguments['restore'][name] = restore_param
        transform = Ghosting(**self.add_base_args(arguments))
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed

    def get_params(
        self,
        num_ghosts_range: tuple[int, int],
        axes: tuple[int, ...],
        intensity_range: tuple[float, float],
        restore_range: Optional[tuple[float, float]],
    ) -> tuple[int, int, float, Optional[float]]:
        ng_min, ng_max = num_ghosts_range
        num_ghosts = int(torch.randint(ng_min, ng_max + 1, (1,)).item())
        axis = axes[torch.randint(0, len(axes), (1,))]
        intensity = self.sample_uniform(*intensity_range)
        if restore_range is None:
            restore = None
        else:
            restore = self.sample_uniform(*restore_range)
        return num_ghosts, axis, intensity, restore


class Ghosting(IntensityTransform, FourierTransform):
    r"""Add MRI ghosting artifact.

    Discrete "ghost" artifacts may occur along the phase-encode direction
    whenever the position or signal intensity of imaged structures within the
    field-of-view vary or move in a regular (periodic) fashion. Pulsatile flow
    of blood or CSF, cardiac motion, and respiratory motion are the most
    important patient-related causes of ghost artifacts in clinical MR imaging
    (from `mriquestions.com`_).

    .. _mriquestions.com: http://mriquestions.com/why-discrete-ghosts.html

    Args:
        num_ghosts: Number of 'ghosts' :math:`n` in the image.
        axes: Axis along which the ghosts will be created.
        intensity: Positive number representing the artifact strength
            :math:`s` with respect to the maximum of the :math:`k`-space.
            If ``0``, the ghosts will not be visible.
        restore: Number between ``0`` and ``1`` indicating how much of the
            :math:`k`-space center should be restored after removing the planes
            that generate the artifact. If ``None``, only the central slice
            will be restored.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. note:: The execution time of this transform does not depend on the
        number of ghosts.
    """

    def __init__(
        self,
        num_ghosts: Union[int, dict[str, int]],
        axis: Union[int, dict[str, int]],
        intensity: Union[float, dict[str, float]],
        restore: Union[Optional[float], dict[str, Optional[float]]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axis = axis
        self.num_ghosts = num_ghosts
        self.intensity = intensity
        self.restore = restore
        self.args_names = ['num_ghosts', 'axis', 'intensity', 'restore']

    def apply_transform(self, subject: Subject) -> Subject:
        axis: Union[int, dict[str, int]]
        num_ghosts: Union[int, dict[str, int]]
        intensity: Union[float, dict[str, float]]
        restore: Union[Optional[float], dict[str, Optional[float]]]
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                assert isinstance(self.axis, dict)
                assert isinstance(self.num_ghosts, dict)
                assert isinstance(self.intensity, dict)
                assert isinstance(self.restore, dict)
                axis = self.axis[name]
                num_ghosts = self.num_ghosts[name]
                intensity = self.intensity[name]
                restore = self.restore[name]
            else:
                axis = self.axis
                num_ghosts = self.num_ghosts
                intensity = self.intensity
                restore = self.restore
            transformed_tensors = []
            for tensor in image.data:
                assert isinstance(num_ghosts, int)
                assert isinstance(axis, int)
                assert isinstance(intensity, (int, float))
                if restore is not None:
                    assert isinstance(restore, float)
                transformed_tensor = self.add_artifact(
                    tensor,
                    num_ghosts,
                    axis,
                    intensity,
                    restore,
                )
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject

    def add_artifact(
        self,
        tensor: torch.Tensor,
        num_ghosts: int,
        axis: int,
        intensity: float,
        restore_center: Optional[float],
    ):
        if not num_ghosts or not intensity:
            return tensor

        spectrum = self.fourier_transform(tensor)

        # Variable "planes" is the part of the spectrum that will be modified
        # Variable "restore" is the part of the spectrum that will be restored
        planes = self._get_planes_to_modify(spectrum, axis, num_ghosts)
        tensor_restore, slices = self._get_slices_to_restore(
            spectrum, axis, restore_center
        )
        tensor_restore = tensor_restore.clone()

        # Multiply by 0 if intensity is 1
        planes *= 1 - intensity

        # Restore the center of k-space to avoid extreme artifacts
        spectrum[slices] = tensor_restore

        tensor_ghosts = self.inv_fourier_transform(spectrum)
        return tensor_ghosts.real.float()

    @staticmethod
    def _get_planes_to_modify(
        spectrum: torch.Tensor,
        axis: int,
        num_ghosts: int,
    ) -> torch.Tensor:
        slices = [slice(None)] * spectrum.ndim
        slices[axis] = slice(None, None, num_ghosts)
        slices_tuple = tuple(slices)
        return spectrum[slices_tuple]

    @staticmethod
    def _get_slices_to_restore(
        spectrum: torch.Tensor,
        axis: int,
        restore_center: Optional[float],
    ) -> tuple[torch.Tensor, tuple[slice, ...]]:
        dim_shape = spectrum.shape[axis]
        mid_idx = dim_shape // 2
        slices = [slice(None)] * spectrum.ndim
        if restore_center is None:
            slice_ = slice(mid_idx, mid_idx + 1)
        else:
            size_restore = int(np.round(restore_center * dim_shape))
            slice_ = slice(mid_idx - size_restore // 2, mid_idx + size_restore // 2)
        slices[axis] = slice_
        slices_tuple = tuple(slices)
        restore_tensor = spectrum[slices_tuple]
        return restore_tensor, slices_tuple
