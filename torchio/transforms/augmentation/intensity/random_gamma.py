from typing import Tuple, Optional, List
import torch
from ....torchio import DATA, TypeRangeFloat
from ....data.subject import Subject
from .. import RandomTransform


class RandomGamma(RandomTransform):
    r"""Change contrast of an image by setting its values to the power
    :math:`\gamma`.

    Args:
        log_gamma_range: Tuple :math:`(a, b)` to compute the factor
            :math:`\gamma` used to set the intensity to the power :math:`gamma`,
            where :math:`\log(\gamma) \sim \mathcal{U}(a, b)`.
            If a single value :math:`d` is provided, then
            :math:`\log(\gamma) \sim \mathcal{U}(-d, d)`.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
        keys: See :py:class:`~torchio.transforms.Transform`.

    Example:
        >>> import torchio
        >>> from torchio import RandomGamma, RescaleIntensity, Compose
        >>> from torchio.datasets import FPG
        >>> sample = FPG()
        >>> gamma_transform = RandomGamma(log_gamma_range=(-0.3, 0.3))
        >>> # It's better to rescale data to [0, 1] to avoid exploding values
        >>> rescale_transform = RescaleIntensity((0, 1), (1, 99))
        >>> transform = Compose([rescale_transform, gamma_transform])
        >>> transformed = transform(sample)
    """
    def __init__(
            self,
            log_gamma_range: TypeRangeFloat = (-0.3, 0.3),
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.log_gamma_range = self.parse_range(log_gamma_range, 'log_gamma')

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        for image_name, image_dict in sample.get_images_dict().items():
            gamma = self.get_params(self.log_gamma_range)
            random_parameters_dict = {'gamma': gamma}
            random_parameters_images_dict[image_name] = random_parameters_dict
            image_dict[DATA] = image_dict[DATA] ** gamma
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    @staticmethod
    def get_params(log_gamma_range: Tuple[float, float]) -> torch.Tensor:
        gamma = torch.FloatTensor(1).uniform_(*log_gamma_range).exp()
        return gamma
