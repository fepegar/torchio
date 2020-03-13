from typing import Union, Callable
import torch
from ....torchio import DATA, TypeCallable
from . import NormalizationTransform


class ZNormalization(NormalizationTransform):
    """Subtract mean and divide by standard deviation.

    Args:
        masking_method: See
            :py:class:`~torchio.transforms.preprocessing.normalization_transform.NormalizationTransform`.
    """
    def __init__(self, masking_method: Union[str, TypeCallable, None] = None):
        super().__init__(masking_method=masking_method)

    def apply_normalization(
            self,
            sample: dict,
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
