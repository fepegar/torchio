import warnings
from typing import Tuple, Optional, List
import torch
from ....torchio import DATA, TypeRangeFloat
from ....data.subject import Subject
from ... import IntensityTransform
from .. import RandomTransform


class RandomGamma(RandomTransform, IntensityTransform):
    r"""Change contrast of an image by raising its values to the power
    :math:`\gamma`.

    Args:
        log_gamma: Tuple :math:`(a, b)` to compute the exponent
            :math:`\gamma = e ^ \beta`,
            where :math:`\beta \sim \mathcal{U}(a, b)`.
            If a single value :math:`d` is provided, then
            :math:`\beta \sim \mathcal{U}(-d, d)`.
            Negative and positive values for this argument perform gamma
            compression and expansion, respectively.
            See the `Gamma correction`_ Wikipedia entry for more information.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
        keys: See :py:class:`~torchio.transforms.Transform`.

    .. _Gamma correction: https://en.wikipedia.org/wiki/Gamma_correction

    .. warning:: Fractional exponentiation of negative values is generally not
        well-defined for non-complex numbers.
        If negative values are found in the input image :math:`I`,
        the applied transform is :math:`\text{sign}(I) |I|^\gamma`,
        instead of the usual :math:`I^\gamma`. The
        :py:class:`~torchio.transforms.preprocessing.intensity.rescale.RescaleIntensity`
        transform may be used to ensure that all values are positive.

    Example:
        >>> import torchio
        >>> from torchio import RandomGamma
        >>> from torchio.datasets import FPG
        >>> sample = FPG()
        >>> transform = RandomGamma(log_gamma=(-0.3, 0.3))  # gamma between 0.74 and 1.34
        >>> transformed = transform(sample)
    """
    def __init__(
            self,
            log_gamma: TypeRangeFloat = (-0.3, 0.3),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.log_gamma_range = self.parse_range(log_gamma, 'log_gamma')

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        for image_name, image_dict in self.get_images_dict(sample).items():
            gamma = self.get_params(self.log_gamma_range)
            random_parameters_dict = {'gamma': gamma}
            random_parameters_images_dict[image_name] = random_parameters_dict
            if torch.any(image_dict[DATA] < 0):
                message = (
                    'Negative values found in input tensor. See the'
                    ' documentation for more details on the implemented'
                    ' workaround:'
                    ' https://torchio.readthedocs.io/transforms/augmentation.html#randomgamma'
                )
                warnings.warn(message)
                data = image_dict[DATA]
                image_dict[DATA] = data.sign() * data.abs() ** gamma
            else:
                image_dict[DATA] = image_dict[DATA] ** gamma
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    @staticmethod
    def get_params(log_gamma_range: Tuple[float, float]) -> torch.Tensor:
        gamma = torch.FloatTensor(1).uniform_(*log_gamma_range).exp()
        return gamma
