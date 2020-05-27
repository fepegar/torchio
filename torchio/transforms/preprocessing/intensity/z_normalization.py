import torch
from ....data.subject import Subject
from ....torchio import DATA
from .normalization_transform import NormalizationTransform, TypeMaskingMethod


class ZNormalization(NormalizationTransform):
    """Subtract mean and divide by standard deviation.

    Args:
        masking_method: See
            :py:class:`~torchio.transforms.preprocessing.normalization_transform.NormalizationTransform`.
        p: Probability that this transform will be applied.
    """
    def __init__(
            self,
            masking_method: TypeMaskingMethod = None,
            p: float = 1,
            ):
        super().__init__(masking_method=masking_method, p=p)

    def apply_normalization(
            self,
            sample: Subject,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        image_dict = sample[image_name]
        image_dict[DATA] = self.znorm(
            image_dict[DATA],
            mask,
        )

    @staticmethod
    def znorm(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = tensor[mask]
        mean, std = values.mean(), values.std()
        tensor = tensor - mean
        tensor = tensor / std
        return tensor
