from collections import defaultdict
from typing import Tuple, Union, Dict

import torch
import numpy as np

from ....data.subject import Subject
from ... import IntensityTransform, FourierTransform
from .. import RandomTransform


class RandomGhosting(RandomTransform, IntensityTransform):
    r"""Add random MRI ghosting artifact.

    Discrete "ghost" artifacts may occur along the phase-encode direction
    whenever the position or signal intensity of imaged structures within the
    field-of-view vary or move in a regular (periodic) fashion. Pulsatile flow
    of blood or CSF, cardiac motion, and respiratory motion are the most
    important patient-related causes of ghost artifacts in clinical MR imaging
    (from `mriquestions.com`_).

    .. _mriquestions.com: http://mriquestions.com/why-discrete-ghosts.html

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
            that generate the artifact.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. note:: The execution time of this transform does not depend on the
        number of ghosts.
    """
    def __init__(
            self,
            num_ghosts: Union[int, Tuple[int, int]] = (4, 10),
            axes: Union[int, Tuple[int, ...]] = (0, 1, 2),
            intensity: Union[float, Tuple[float, float]] = (0.5, 1),
            restore: float = 0.02,
            **kwargs
            ):
        super().__init__(**kwargs)
        if not isinstance(axes, tuple):
            try:
                axes = tuple(axes)
            except TypeError:
                axes = (axes,)
        for axis in axes:
            if not isinstance(axis, str) and axis not in (0, 1, 2):
                raise ValueError(f'Axes must be in (0, 1, 2), not "{axes}"')
        self.axes = axes
        self.num_ghosts_range = self._parse_range(
            num_ghosts, 'num_ghosts', min_constraint=0, type_constraint=int)
        self.intensity_range = self._parse_range(
            intensity, 'intensity_range', min_constraint=0)
        self.restore = _parse_restore(restore)

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        if any(isinstance(n, str) for n in self.axes):
            subject.check_consistent_orientation()
        for name, image in self.get_images_dict(subject).items():
            is_2d = image.is_2d()
            axes = [a for a in self.axes if a != 2] if is_2d else self.axes
            params = self.get_params(
                self.num_ghosts_range,
                axes,
                self.intensity_range,
            )
            num_ghosts_param, axis_param, intensity_param = params
            arguments['num_ghosts'][name] = num_ghosts_param
            arguments['axis'][name] = axis_param
            arguments['intensity'][name] = intensity_param
            arguments['restore'][name] = self.restore
        transform = Ghosting(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(
            self,
            num_ghosts_range: Tuple[int, int],
            axes: Tuple[int, ...],
            intensity_range: Tuple[float, float],
            ) -> Tuple:
        ng_min, ng_max = num_ghosts_range
        num_ghosts = torch.randint(ng_min, ng_max + 1, (1,)).item()
        axis = axes[torch.randint(0, len(axes), (1,))]
        intensity = self.sample_uniform(*intensity_range).item()
        return num_ghosts, axis, intensity


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
            that generate the artifact.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. note:: The execution time of this transform does not depend on the
        number of ghosts.
    """
    def __init__(
            self,
            num_ghosts: Union[int, Dict[str, int]],
            axis: Union[int, Dict[str, int]],
            intensity: Union[float, Dict[str, float]],
            restore: Union[float, Dict[str, float]],
            **kwargs
            ):
        super().__init__(**kwargs)
        self.axis = axis
        self.num_ghosts = num_ghosts
        self.intensity = intensity
        self.restore = restore
        self.args_names = 'num_ghosts', 'axis', 'intensity', 'restore'

    def apply_transform(self, subject: Subject) -> Subject:
        axis = self.axis
        num_ghosts = self.num_ghosts
        intensity = self.intensity
        restore = self.restore
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                axis = self.axis[name]
                num_ghosts = self.num_ghosts[name]
                intensity = self.intensity[name]
                restore = self.restore[name]
            transformed_tensors = []
            for tensor in image.data:
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
            restore_center: float,
            ):
        if not num_ghosts or not intensity:
            return tensor

        array = tensor.numpy()
        spectrum = self.fourier_transform(array)

        mi, mj, mk = np.array(array.shape) // 2

        # Variable "planes" is the part of the spectrum that will be modified
        if axis == 0:
            planes = spectrum[::num_ghosts, :, :]
            restore = spectrum[mi, :, :].copy()
        elif axis == 1:
            planes = spectrum[:, ::num_ghosts, :]
            restore = spectrum[:, mj, :].copy()
        elif axis == 2:
            planes = spectrum[:, :, ::num_ghosts]
            restore = spectrum[:, :, mk].copy()

        # Multiply by 0 if intensity is 1
        planes *= 1 - intensity

        # Restore the center of k-space to avoid extreme artifacts
        if axis == 0:
            spectrum[mi, :, :] = restore
        elif axis == 1:
            spectrum[:, mj, :] = restore
        elif axis == 2:
            spectrum[:, :, mk] = restore

        array_ghosts = self.inv_fourier_transform(spectrum)
        array_ghosts = np.real(array_ghosts).astype(np.float32)
        return torch.as_tensor(array_ghosts)


def _parse_restore(restore):
    try:
        restore = float(restore)
    except Exception as e:
        raise TypeError(f'Restore must be a float, not "{restore}"') from e
    if not 0 <= restore <= 1:
        message = (
            f'Restore must be a number between 0 and 1, not {restore}')
        raise ValueError(message)
    return restore
