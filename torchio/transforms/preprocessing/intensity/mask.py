from typing import Optional, Sequence

import torch
import numpy as np

from ....data.image import LabelMap
from ....data.subject import Subject
from ....transforms.transform import TypeMaskingMethod
from ... import IntensityTransform


class Mask(IntensityTransform):
    """Set voxels outside of mask to a constant value.

    Args:
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        outside_value: Value to set for all voxels outside of the mask.
        labels: If a label map is used to generate the mask,
            sequence of labels to consider.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> subject = tio.datasets.Colin27()
        >>> subject
        Colin27(Keys: ('t1', 'head', 'brain'); images: 3)
        >>> mask = tio.Mask(masking_method='brain')  # Use "brain" image to mask
        >>> transformed = mask(subject)  # Set values outside of the brain to 0
    """  # noqa: E501
    def __init__(
            self,
            masking_method: TypeMaskingMethod,
            outside_value: float = 0,
            labels: Optional[Sequence[int]] = None,
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.masking_method = masking_method
        self.masking_labels = labels
        self.outside_value = outside_value
        self.args_names = ('masking_method',)

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            label_map = self.get_mask_from_masking_method(
                self.masking_method,
                subject,
                image.data,
            )
            self.apply_masking(image, label_map)
        return subject

    def apply_masking(self, image: LabelMap, label_map: torch.Tensor) -> None:
        masked = self.mask(image.data, label_map, self.masking_labels)
        image.set_data(masked)

    def mask(
            self,
            tensor: torch.Tensor,
            label_map: torch.Tensor,
            labels: Optional[Sequence[int]] = None,
            ) -> torch.Tensor:
        array = tensor.clone().numpy()
        label_map = label_map.numpy()
        mask = label_map if labels is None else np.isin(label_map, labels)
        array[~mask] = self.outside_value
        return torch.as_tensor(array)
