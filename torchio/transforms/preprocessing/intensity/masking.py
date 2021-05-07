from typing import Optional, List
import torch
from ....data.subject import Subject
from ....transforms.transform import TypeMaskingMethod
from ... import IntensityTransform
import numpy as np


class Mask(IntensityTransform):
    """Set intensity values outside of mask to a constant value.

    Args:
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        outside_mask_value: Value to set for all voxels outside of the mask.
        masking_labels: Labels from the LabelMask to consider as for the mask.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> subject = tio.datasets.Colin27()
        >>> subject
        Colin27(Keys: ('t1', 'head', 'brain'); images: 3)
        >>> transform = tio.Mask(masking_method='brain')  # Set brain as mask
        >>> transformed = transform(subject)  # Set values outside of mask to 0
    """
    def __init__(
            self,
            masking_method: TypeMaskingMethod,
            outside_mask_value: Optional[int] = 0,
            masking_labels: Optional[List[int]] = None,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.masking_method = masking_method
        self.masking_labels = masking_labels
        self.outside_mask_value = outside_mask_value
        self.args_names = ('masking_method',)

    def apply_transform(self, subject: Subject) -> Subject:
        for image_name, image in self.get_images_dict(subject).items():
            mask = self.get_mask_from_masking_method(
                self.masking_method,
                subject,
                image.data,
            )
            self.apply_masking(subject, image_name, mask)
        return subject

    def apply_masking(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        image = subject[image_name]
        masked = self.mask(
            image.data,
            mask,
        )
        image.set_data(masked)

    def mask(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        array = tensor.clone().float().numpy()
        mask = mask.numpy()

        if self.masking_labels is None:
            array[~mask] = self.outside_mask_value
        else:
            masked_elements = np.invert(np.isin(mask, self.masking_labels))
            array[masked_elements] = self.outside_mask_value
        return torch.as_tensor(array)
