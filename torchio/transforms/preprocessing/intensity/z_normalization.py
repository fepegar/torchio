from typing import Union, Optional, Callable
import torch
from ....torchio import DATA, TypeCallable
from .normalization_transform import NormalizationTransform


class ZNormalization(NormalizationTransform):
    """
    Subtract mean and divide by standard deviation
    """
    def __init__(
            self,
            masking_method: Optional[Union[str, TypeCallable]] = None,
            verbose: bool = False,
            ):
        super().__init__(masking_method=masking_method, verbose=verbose)

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
