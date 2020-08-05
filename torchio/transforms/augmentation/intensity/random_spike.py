from typing import Tuple, Optional, Union, List
import torch
import numpy as np
import SimpleITK as sitk
from ....torchio import DATA, AFFINE
from ....data.subject import Subject
from .. import RandomTransform


class RandomSpike(RandomTransform):
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
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
        keys: See :py:class:`~torchio.transforms.Transform`.

    .. note:: The execution time of this transform does not depend on the
        number of spikes.
    """
    def __init__(
            self,
            num_spikes: Union[int, Tuple[int, int]] = 1,
            intensity: Union[float, Tuple[float, float]] = (1, 3),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.intensity_range = self.parse_range(
            intensity, 'intensity_range')
        self.num_spikes_range = self.parse_range(
            num_spikes, 'num_spikes', min_constraint=0, type_constraint=int)

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        for image_name, image in sample.get_images_dict().items():
            transformed_tensors = []
            for channel_idx, channel in enumerate(image[DATA]):
                params = self.get_params(
                    self.num_spikes_range,
                    self.intensity_range,
                )
                spikes_positions_param, intensity_param = params
                random_parameters_dict = {
                    'intensity': intensity_param,
                    'spikes_positions': spikes_positions_param,
                }
                key = f'{image_name}_channel_{channel_idx}'
                random_parameters_images_dict[key] = random_parameters_dict
                transformed_tensor = self.add_artifact(
                    channel,
                    spikes_positions_param,
                    intensity_param,
                )
                transformed_tensors.append(transformed_tensor)
            image[DATA] = torch.stack(transformed_tensors)
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    @staticmethod
    def get_params(
            num_spikes_range: Tuple[int, int],
            intensity_range: Tuple[float, float],
            ) -> Tuple:
        ns_min, ns_max = num_spikes_range
        num_spikes_param = torch.randint(ns_min, ns_max + 1, (1,)).item()
        intensity_param = torch.FloatTensor(1).uniform_(*intensity_range)
        spikes_positions = torch.rand(num_spikes_param, 3).numpy()
        return spikes_positions, intensity_param.item()

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
            spectrum[i, j, k] += spectrum.max() * intensity_factor
            # If we wanted to add a pure cosine, we should add spikes to both
            # sides of k-space. However, having only one is a better
            # representation og the actual cause of the artifact in real
            # scans.
            #i, j, k = mid_shape - diff
            #spectrum[i, j, k] = spectrum.max() * intensity_factor
        result = np.real(self.inv_fourier_transform(spectrum))
        return torch.from_numpy(result.astype(np.float32))
