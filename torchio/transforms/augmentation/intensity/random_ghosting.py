from typing import Tuple, Optional, Union, List
import torch
import numpy as np
from ....torchio import DATA
from ....data.subject import Subject
from ... import IntensityTransform
from .. import RandomTransform


class RandomGhosting(RandomTransform, IntensityTransform):
    r"""Add random MRI ghosting artifact.

    Discrete "ghost" artifacts may occur along the phase-encode direction
    whenever the position or signal intensity of imaged structures within the
    field-of-view vary or move in a regular (periodic) fashion. Pulsatile flow
    of blood or CSF, cardiac motion, and respiratory motion are the most
    important patient-related causes of ghost artifacts in clinical MR imaging
    (from `mriquestions.com <http://mriquestions.com/why-discrete-ghosts.html>`_).

    Args:
        num_ghosts: Number of 'ghosts' :math:`n` in the image.
            If :py:attr:`num_ghosts` is a tuple :math:`(a, b)`, then
            :math:`n \sim \mathcal{U}(a, b) \cap \mathbb{N}`.
            If only one value :math:`d` is provided,
            :math:`n \sim \mathcal{U}(0, d) \cap \mathbb{N}`.
        axes: Axis along which the ghosts will be created. If
            :py:attr:`axes` is a tuple, the axis will be randomly chosen
            from the passed values. Anatomical labels may also be used (see
            :py:class:`~torchio.transforms.augmentation.RandomFlip`).
        intensity: Positive number representing the artifact strength
            :math:`s` with respect to the maximum of the :math:`k`-space.
            If ``0``, the ghosts will not be visible. If a tuple
            :math:`(a, b)` is provided then :math:`s \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`s \sim \mathcal{U}(0, d)`.
        restore: Number between ``0`` and ``1`` indicating how much of the
            :math:`k`-space center should be restored after removing the planes
            that generate the artifact.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
        keys: See :py:class:`~torchio.transforms.Transform`.

    .. note:: The execution time of this transform does not depend on the
        number of ghosts.

    .. warning:: Note that height and width of 2D images correspond to axes
        ``1`` and ``2`` respectively, as TorchIO images are generally considered
        to have 3 spatial dimensions.
    """
    def __init__(
            self,
            num_ghosts: Union[int, Tuple[int, int]] = (4, 10),
            axes: Union[int, Tuple[int, ...]] = (0, 1, 2),
            intensity: Union[float, Tuple[float, float]] = (0.5, 1),
            restore: float = 0.02,
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, seed=seed, keys=keys)
        if not isinstance(axes, tuple):
            try:
                axes = tuple(axes)
            except TypeError:
                axes = (axes,)
        for axis in axes:
            if not isinstance(axis, str) and axis not in (0, 1, 2):
                raise ValueError(f'Axes must be in (0, 1, 2), not "{axes}"')
        self.axes = axes
        self.num_ghosts_range = self.parse_range(
            num_ghosts, 'num_ghosts', min_constraint=0, type_constraint=int)
        self.intensity_range = self.parse_range(
            intensity, 'intensity_range', min_constraint=0)
        self.restore = self.parse_restore(restore)

    @staticmethod
    def parse_restore(restore):
        if not isinstance(restore, float):
            raise TypeError(f'Restore must be a float, not {restore}')
        if not 0 <= restore <= 1:
            message = (
                f'Restore must be a number between 0 and 1, not {restore}')
            raise ValueError(message)
        return restore

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        if any(isinstance(n, str) for n in self.axes):
            sample.check_consistent_orientation()
        for image_name, image in self.get_images_dict(sample).items():
            transformed_tensors = []
            is_2d = image.is_2d()
            axes = [a for a in self.axes if a != 2] if is_2d else self.axes
            for channel_idx, tensor in enumerate(image[DATA]):
                params = self.get_params(
                    self.num_ghosts_range,
                    axes,
                    self.intensity_range,
                )
                num_ghosts_param, axis_param, intensity_param = params
                random_parameters_dict = {
                    'axis': axis_param,
                    'num_ghosts': num_ghosts_param,
                    'intensity': intensity_param,
                }
                key = f'{image_name}_channel_{channel_idx}'
                random_parameters_images_dict[key] = random_parameters_dict
                transformed_tensor = self.add_artifact(
                    tensor,
                    num_ghosts_param,
                    axis_param,
                    intensity_param,
                    self.restore,
                )
                transformed_tensors.append(transformed_tensor)
            image[DATA] = torch.stack(transformed_tensors)
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    @staticmethod
    def get_params(
            num_ghosts_range: Tuple[int, int],
            axes: Tuple[int, ...],
            intensity_range: Tuple[float, float],
            ) -> Tuple:
        ng_min, ng_max = num_ghosts_range
        num_ghosts = torch.randint(ng_min, ng_max + 1, (1,)).item()
        axis = axes[torch.randint(0, len(axes), (1,))]
        intensity = torch.FloatTensor(1).uniform_(*intensity_range).item()
        return num_ghosts, axis, intensity

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

        shape = np.array(array.shape)
        ri, rj, rk = np.round(restore_center * shape).astype(np.uint16)
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
        array_ghosts = np.real(array_ghosts)
        return torch.from_numpy(array_ghosts)
