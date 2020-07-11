from typing import Tuple, Optional, Union
import torch
import numpy as np
import SimpleITK as sitk
from ....torchio import DATA, AFFINE
from ....data.subject import Subject
from .. import RandomTransform


class RandomGhosting(RandomTransform):
    r"""Add random MRI ghosting artifact.

    Args:
        num_ghosts: Number of 'ghosts' :math:`n` in the image.
            If :py:attr:`num_ghosts` is a tuple :math:`(a, b)`, then
            :math:`n \sim \mathcal{U}(a, b) \cap \mathbb{N}`.
        axes: Axis along which the ghosts will be created. If
            :py:attr:`axes` is a tuple, the axis will be randomly chosen
            from the passed values.
        intensity: Positive number representing the artifact strength
            :math:`s` with respect to the maximum of the :math:`k`-space.
            If ``0``, the ghosts will not be visible. If a tuple
            :math:`(a, b)` is provided then :math:`s \sim \mathcal{U}(a, b)`.
        restore: Number between ``0`` and ``1`` indicating how much of the
            :math:`k`-space center should be restored after removing the planes
            that generate the artifact.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.

    .. note:: The execution time of this transform does not depend on the
        number of ghosts.
    """
    def __init__(
            self,
            num_ghosts: Union[int, Tuple[int, int]] = (4, 10),
            axes: Union[int, Tuple[int, ...]] = (0, 1, 2),
            intensity: Union[float, Tuple[float, float]] = (0.5, 1),
            restore: float = 0.02,
            p: float = 1,
            seed: Optional[int] = None,
            ):
        super().__init__(p=p, seed=seed)
        if not isinstance(axes, tuple):
            try:
                axes = tuple(axes)
            except TypeError:
                axes = (axes,)
        for axis in axes:
            if axis not in (0, 1, 2):
                raise ValueError(f'Axes must be in (0, 1, 2), not "{axes}"')
        self.axes = axes
        self.num_ghosts_range = self.parse_num_ghosts(num_ghosts)
        self.intensity_range = self.parse_intensity(intensity)
        if not 0 <= restore < 1:
            message = (
                f'Restore must be a number between 0 and 1, not {restore}')
            raise ValueError(message)
        self.restore = restore

    @staticmethod
    def parse_num_ghosts(num_ghosts):
        try:
            iter(num_ghosts)
        except TypeError:
            num_ghosts = num_ghosts, num_ghosts
        for n in num_ghosts:
            if not isinstance(n, int) or n < 0:
                message = (
                    f'Number of ghosts must be a natural number, not {n}')
                raise ValueError(message)
        return num_ghosts

    @staticmethod
    def parse_intensity(intensity):
        try:
            iter(intensity)
        except TypeError:
            intensity = intensity, intensity
        for n in intensity:
            if n < 0:
                message = (
                    f'Intensity must be a positive number, not {n}')
                raise ValueError(message)
        return intensity

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        for image_name, image_dict in sample.get_images_dict().items():
            data = image_dict[DATA]
            is_2d = data.shape[-3] == 1
            axes = [a for a in self.axes if a != 0] if is_2d else self.axes
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
            random_parameters_images_dict[image_name] = random_parameters_dict
            image_dict[DATA][0] = self.add_artifact(
                data[0],
                num_ghosts_param,
                axis_param,
                intensity_param,
                self.restore,
            )
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
        array = tensor.numpy()
        spectrum = self.fourier_transform(array)

        ri, rj, rk = np.round(restore_center * np.array(array.shape)).astype(np.uint16)
        mi, mj, mk = np.array(array.shape) // 2

        # Variable "planes" is the part the spectrum that will be modified
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
