import torch

from ...transform import TypeMaskingMethod
from .remap_labels import RemapLabels
from .label_transform import LabelTransform


class SequentialLabels(LabelTransform):
    r"""Remap the integer IDs of labels in a LabelMap to be sequential.

    For example, if a label map has 6 labels with IDs (3, 5, 9, 15, 16, 23),
    then this will apply a :class:`~torchio.RemapLabels` transform with
    ``remapping={3: 1, 5: 2, 9: 3, 15: 4, 16: 5, 23: 6}``.
    This transformation is always `fully invertible <invertibility>`_.

    Args:
        masking_method: See :class:`~torchio.transforms.RemapLabels`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            masking_method: TypeMaskingMethod = None,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.masking_method = masking_method
        self.args_names = []

    def apply_transform(self, subject):
        for name, image in self.get_images_dict(subject).items():
            unique_labels = torch.unique(image.data)
            remapping = {
                unique_labels[i].item(): i
                for i in range(1, len(unique_labels))
            }
            transform = RemapLabels(
                remapping=remapping,
                masking_method=self.masking_method,
                include=name,
            )
            subject = transform(subject)

        return subject
