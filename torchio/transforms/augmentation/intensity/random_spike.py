import warnings
from typing import Tuple, Optional, Union
import torch
import numpy as np
import SimpleITK as sitk
from ....torchio import DATA, AFFINE
from ....data.subject import Subject
from .. import RandomTransform


class RandomSpike(RandomTransform):
    r"""Add random MRI spike artifacts.

    Args:
        num_spikes: Number of spikes :math:`n` presnet in k-space.
            If a tuple :math:`(a, b)` is provided, then
            :math:`n \sim \mathcal{U}(a, b) \cap \mathbb{N}`.
            Larger values generate more distorted images.
        intensity: Ratio :math:`r` between the spike intensity and the maximum
            of the spectrum.
            Larger values generate more distorted images.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.

    .. note:: The execution time of this transform does not depend on the
        number of spikes.
    """
    def __init__(
            self,
            num_spikes: Union[int, Tuple[int, int]] = 1,
            intensity: Union[float, Tuple[float, float]] = (0.1, 1),
            p: float = 1,
            seed: Optional[int] = None,
            ):
        super().__init__(p=p, seed=seed)
        self.intensity_range = self.parse_range(
            intensity, 'intensity_range')
        if isinstance(num_spikes, int):
            self.num_spikes_range = num_spikes, num_spikes
        else:
            self.num_spikes_range = num_spikes

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        for image_name, image_dict in sample.get_images_dict().items():
            params = self.get_params(
                self.num_spikes_range,
                self.intensity_range,
            )
            spikes_positions_param, intensity_param = params
            random_parameters_dict = {
                'intensity': intensity_param,
                'spikes_positions': spikes_positions_param,
            }
            random_parameters_images_dict[image_name] = random_parameters_dict
            if (image_dict[DATA][0] < -0.1).any():
                # I use -0.1 instead of 0 because Python was warning me when
                # a value in a voxel was -7.191084e-35
                # There must be a better way of solving this
                message = (
                    f'Image "{image_name}" from "{image_dict["stem"]}"'
                    ' has negative values.'
                    ' Results can be unexpected because the transformed sample'
                    ' is computed as the absolute values'
                    ' of an inverse Fourier transform'
                )
                warnings.warn(message)
            image = self.nib_to_sitk(
                image_dict[DATA][0],
                image_dict[AFFINE],
            )
            image_dict[DATA] = self.add_artifact(
                image,
                spikes_positions_param,
                intensity_param,
            )
            # Add channels dimension
            image_dict[DATA] = image_dict[DATA][np.newaxis, ...]
            image_dict[DATA] = torch.from_numpy(image_dict[DATA])
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
        spikes_positions = torch.rand(num_spikes_param).numpy()
        return spikes_positions, intensity_param.item()

    def add_artifact(
            self,
            image: sitk.Image,
            spikes_positions: np.ndarray,
            intensity_factor: float,
            ):
        array = sitk.GetArrayViewFromImage(image).transpose()
        spectrum = self.fourier_transform(array).ravel()
        indices = np.floor(spikes_positions * len(spectrum)).astype(int)
        for index in indices:
            spectrum[index] = spectrum.max() * intensity_factor
        spectrum = spectrum.reshape(array.shape)
        result = self.inv_fourier_transform(spectrum)
        return result.astype(np.float32)
