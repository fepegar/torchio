from typing import Union, Sequence, Optional
import torch
from ....data.subject import Subject
from ....torchio import DATA, TypeCallable
from ... import IntensityTransform


TypeMaskingMethod = Union[str, TypeCallable, None]


class NormalizationTransform(IntensityTransform):
    """Base class for intensity preprocessing transforms.

    Args:
        masking_method: Defines the mask used to compute the normalization statistics. It can be one of:

            - ``None``: the mask image is all ones, i.e. all values in the image are used

            - A string: the mask image is retrieved from the subject, which is expected the string as a key

            - A function: the mask image is computed as a function of the intensity image. The function must receive and return a :class:`torch.Tensor`
        keys: See :class:`~torchio.transforms.Transform`.

    Example:
        >>> import torchio as tio
        >>> subject = tio.datasets.Colin27()
        >>> subject
        Colin27(Keys: ('t1', 'head', 'brain'); images: 3)
        >>> transform = tio.ZNormalization()  # ZNormalization is a subclass of NormalizationTransform
        >>> transformed = transform(subject)  # use all values to compute mean and std
        >>> transform = tio.ZNormalization(masking_method='brain')
        >>> transformed = transform(subject)  # use only values within the brain
        >>> transform = tio.ZNormalization(masking_method=lambda x: x > x.mean())
        >>> transformed = transform(subject)  # use values above the image mean

    """
    def __init__(
            self,
            masking_method: TypeMaskingMethod = None,
            p: float = 1,
            keys: Optional[Sequence[str]] = None,
            ):
        """
        masking_method is used to choose the values used for normalization.
        It can be:
         - A string: the mask will be retrieved from the subject
         - A function: the mask will be computed using the function
         - None: all values are used
        """
        super().__init__(p=p, keys=keys)
        self.mask_name = None
        self.masking_method = masking_method
        if masking_method is None:
            self.masking_method = self.ones
        elif callable(masking_method):
            self.masking_method = masking_method
        elif isinstance(masking_method, str):
            self.mask_name = masking_method

    def get_mask(self, subject: Subject, tensor: torch.Tensor) -> torch.Tensor:
        if self.mask_name is None:
            return self.masking_method(tensor)
        else:
            return subject[self.mask_name][DATA].bool()

    def apply_transform(self, subject: Subject) -> Subject:
        for image_name, image_dict in self.get_images_dict(subject).items():
            mask = self.get_mask(subject, image_dict[DATA])
            self.apply_normalization(subject, image_name, mask)
        return subject

    def apply_normalization(
            self,
            subject: Subject,
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
