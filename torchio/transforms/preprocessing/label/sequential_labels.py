import torch

from ....data import LabelMap
from ...transform import Transform, TypeMaskingMethod
from .remap_labels import RemapLabels


class SequentialLabels(Transform):
    r"""Remap the integer IDs of labels in a LabelMap to be sequential.

    For example, if a label map has 6 labels with IDs (3, 5, 9, 15, 16, 23),
    then this will apply a :class:`~torchio.RemapLabels` transform with
    ``remapping={3: 1, 5: 2, 9: 3, 15: 4, 16: 5, 23: 6}``.
    This transformation is always `fully invertible <invertibility>`_.

    Args:
        masking_method: See
            :class:`~torchio.RemapLabels`.
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
        images_dict = subject.get_images_dict(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        for name, image in images_dict.items():
            if not isinstance(image, LabelMap):
                continue

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
