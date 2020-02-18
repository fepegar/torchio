from typing import Union, Optional, Callable
import torch
from ....utils import is_image_dict
from ....torchio import DATA, INTENSITY, TypeCallable
from ... import Transform


class NormalizationTransform(Transform):
    def __init__(
            self,
            masking_method: Optional[Union[str, TypeCallable]] = None,
            verbose: bool = False,
            ):
        """
        masking_method is used to choose the values used for normalization.
        It can be:
         - A string: the mask will be retrieved from the sample
         - A function: the mask will be computed using the function
         - None: all values are used
        """
        super().__init__(verbose=verbose)
        self.mask_name = None
        if masking_method is None:
            self.masking_method = self.ones
        elif callable(masking_method):
            self.masking_method = masking_method
        elif isinstance(masking_method, str):
            self.mask_name = masking_method

    def get_mask(self, sample: dict, tensor: torch.Tensor) -> torch.Tensor:
        if self.mask_name is None:
            return self.masking_method(tensor)
        else:
            return sample[self.mask_name][DATA].bool()

    def apply_transform(self, sample: dict) -> dict:
        for image_name, image_dict in sample.items():
            if not is_image_dict(image_dict):
                continue
            if not image_dict['type'] == INTENSITY:
                continue
            mask = self.get_mask(sample, image_dict[DATA])
            self.apply_normalization(sample, image_name, mask)
        return sample

    def apply_normalization(
            self,
            sample: dict,
            image_name: str,
            mask: torch.Tensor,
            ):
        """There must be a nicer way of doing this"""
        raise NotImplementedError

    @staticmethod
    def ones(tensor: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(tensor, dtype=torch.bool)

    @staticmethod
    def mean(tensor: torch.Tensor) -> torch.Tensor:
        mask = tensor > tensor.mean()
        return mask
