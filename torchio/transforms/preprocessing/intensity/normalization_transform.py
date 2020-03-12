from typing import Union, Callable
import torch
from ....utils import is_image_dict
from ....torchio import DATA, TYPE, INTENSITY, TypeCallable
from ... import Transform


class NormalizationTransform(Transform):
    """Base class for intensity preprocessing transforms.

    Args:
        masking_method: Defines the mask used to compute the normalization statistics. It can be one of:

            - ``None``: the mask image is all ones, i.e. all values in the image are used

            - A string: the mask image is retrieved from the sample, which is expected the string as a key

            - A function: the mask image is computed as a function of the intensity image. The function must receive and return a :py:class:`torch.Tensor`

    Example:
        >>> from torchio.datasets import IXITiny
        >>> from torchio.transforms import ZNormalization
        >>> dataset = IXITiny('ixi_root', download=True)
        >>> sample = dataset[0]
        >>> sample.keys()  # image is the MRI, label is a brain segmentation
        dict_keys(['image', 'label'])
        >>> transform = ZNormalization()  # ZNormalization is a subclass of NormalizationTransform
        >>> transformed = transform(sample)  # use all values to compute mean and std
        >>> transform = ZNormalization(masking_method='label')
        >>> transformed = transform(sample)  # use only values within the brain
        >>> transform = ZNormalization(masking_method=lambda x: x > x.mean())
        >>> transformed = transform(sample)  # use values above the image mean

    """
    def __init__(
            self,
            masking_method: Union[str, TypeCallable, None] = None,
            ):
        """
        masking_method is used to choose the values used for normalization.
        It can be:
         - A string: the mask will be retrieved from the sample
         - A function: the mask will be computed using the function
         - None: all values are used
        """
        super().__init__()
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
            if not image_dict[TYPE] == INTENSITY:
                continue
            mask = self.get_mask(sample, image_dict[DATA])
            self.apply_normalization(sample, image_name, mask)
        return sample

    def apply_normalization(
            self,
            sample: dict,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        # There must be a nicer way of doing this
        raise NotImplementedError

    @staticmethod
    def ones(tensor: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(tensor, dtype=torch.bool)

    @staticmethod
    def mean(tensor: torch.Tensor) -> torch.Tensor:
        mask = tensor > tensor.mean()
        return mask
