from collections import defaultdict
from typing import Tuple, Union, Dict

import torch
import numpy as np

from ....data.subject import Subject
from ... import IntensityTransform, FourierTransform
from .. import RandomTransform


class RandomSpike(RandomTransform, IntensityTransform, FourierTransform):
    r"""Add random MRI spike artifacts.

    Also known as `Herringbone artifact
    <https://radiopaedia.org/articles/herringbone-artifact?lang=gb>`_,
    crisscross artifact or corduroy artifact, it creates stripes in different
    directions in image space due to spikes in k-space.

    Args:
        num_spikes: Number of spikes :math:`n` present in k-space.
            If a tuple :math:`(a, b)` is provided, then
            :math:`n \sim \mathcal{U}(a, b) \cap \mathbb{N}`.
            If only one value :math:`d` is provided,
            :math:`n \sim \mathcal{U}(0, d) \cap \mathbb{N}`.
            Larger values generate more distorted images.
        intensity: Ratio :math:`r` between the spike intensity and the maximum
            of the spectrum.
            If a tuple :math:`(a, b)` is provided, then
            :math:`r \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`r \sim \mathcal{U}(-d, d)`.
            Larger values generate more distorted images.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. note:: The execution time of this transform does not depend on the
        number of spikes.
    """
    def __init__(
            self,
            num_spikes: Union[int, Tuple[int, int]] = 1,
            intensity: Union[float, Tuple[float, float]] = (1, 3),
            **kwargs
            ):
        super().__init__(**kwargs)
        self.intensity_range = self._parse_range(
            intensity, 'intensity_range')
        self.num_spikes_range = self._parse_range(
            num_spikes, 'num_spikes', min_constraint=0, type_constraint=int)

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        for image_name in self.get_images_dict(subject):
            spikes_positions_param, intensity_param = self.get_params(
                self.num_spikes_range,
                self.intensity_range,
            )
            arguments['spikes_positions'][image_name] = spikes_positions_param
            arguments['intensity'][image_name] = intensity_param
        transform = Spike(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(
            self,
            num_spikes_range: Tuple[int, int],
            intensity_range: Tuple[float, float],
            ) -> Tuple[np.ndarray, float]:
        ns_min, ns_max = num_spikes_range
        num_spikes_param = torch.randint(ns_min, ns_max + 1, (1,)).item()
        intensity_param = self.sample_uniform(*intensity_range)
        spikes_positions = torch.rand(num_spikes_param, 3).numpy()
        return spikes_positions, intensity_param.item()


class Spike(IntensityTransform, FourierTransform):
    r"""Add MRI spike artifacts.

    Also known as `Herringbone artifact
    <https://radiopaedia.org/articles/herringbone-artifact?lang=gb>`_,
    crisscross artifact or corduroy artifact, it creates stripes in different
    directions in image space due to spikes in k-space.

    Args:
        spikes_positions:
        intensity: Ratio :math:`r` between the spike intensity and the maximum
            of the spectrum.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. note:: The execution time of this transform does not depend on the
        number of spikes.
    """
    def __init__(
            self,
            spikes_positions: Union[np.ndarray, Dict[str, np.ndarray]],
            intensity: Union[float, Dict[str, float]],
            **kwargs
            ):
        super().__init__(**kwargs)
        self.spikes_positions = spikes_positions
        self.intensity = intensity
        self.args_names = 'spikes_positions', 'intensity'
        self.invert_transform = False

    def apply_transform(self, subject: Subject) -> Subject:
        spikes_positions = self.spikes_positions
        intensity = self.intensity
        for image_name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                spikes_positions = self.spikes_positions[image_name]
                intensity = self.intensity[image_name]
            transformed_tensors = []
            for channel in image.data:
                transformed_tensor = self.add_artifact(
                    channel,
                    spikes_positions,
                    intensity,
                )
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject

    def add_artifact(
            self,
            tensor: torch.Tensor,
            spikes_positions: np.ndarray,
            intensity_factor: float,
            ):
        array = np.asarray(tensor)
        spectrum = self.fourier_transform(array)
        shape = np.array(spectrum.shape)
        mid_shape = shape // 2
        indices = np.floor(spikes_positions * shape).astype(int)
        for index in indices:
            diff = index - mid_shape
            i, j, k = mid_shape + diff
            artifact = spectrum.max() * intensity_factor
            if self.invert_transform:
                spectrum[i, j, k] -= artifact
            else:
                spectrum[i, j, k] += artifact
            # If we wanted to add a pure cosine, we should add spikes to both
            # sides of k-space. However, having only one is a better
            # representation og the actual cause of the artifact in real
            # scans. Therefore the next two lines have been removed.
            # #i, j, k = mid_shape - diff
            # #spectrum[i, j, k] = spectrum.max() * intensity_factor
        result = np.real(self.inv_fourier_transform(spectrum))
        return torch.as_tensor(result.astype(np.float32))
