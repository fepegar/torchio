import torch

from ...transform import TypeMaskingMethod
from .label_transform import LabelTransform
from .remap_labels import RemapLabels


class SequentialLabels(LabelTransform):
    r"""Remap labels in a label map so they become consecutive.

    For example, if a label map has labels ``(0, 3, 5)``, then this will apply
    a :class:`~torchio.RemapLabels` transform with ``remapping={3: 1, 5: 2}``,
    and therefore the output image will have labels ``(0, 1, 2)``.

    Example:

        >>> import torch
        >>> import torchio as tio
        >>> def get_image(*labels):
        ...     tensor = torch.as_tensor(labels).reshape(1, 1, 1, -1)
        ...     image = tio.LabelMap(tensor=tensor)
        ...     return image
        ...
        >>> img_with_bg = get_image(0, 5, 10)
        >>> transform = tio.SequentialLabels()
        >>> transform(img_with_bg).data
        tensor([[[[0, 1, 2]]]])
        >>> img_without_bg = get_image(7, 11, 99)
        >>> transform(img_without_bg).data
        tensor([[[[0, 1, 2]]]])

    .. note::
        This transformation is always `fully invertible <invertibility>`_.

    .. warning::
        The background is typically represented with the label ``0``. There
        will be zeros in the output image even if they are none in the input.

    Args:
        masking_method: See :class:`~torchio.transforms.RemapLabels`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, masking_method: TypeMaskingMethod = None, **kwargs):
        super().__init__(**kwargs)
        self.masking_method = masking_method

    def apply_transform(self, subject):
        for name, image in self.get_images_dict(subject).items():
            unique_labels = torch.unique(image.data)
            remapping = {
                unique_labels[i].item(): i for i in range(0, len(unique_labels))
            }
            init_kwargs = self.get_base_args()
            init_kwargs['include'] = [name]

            transform = RemapLabels(
                remapping=remapping,
                masking_method=self.masking_method,
                **init_kwargs,
            )
            subject = transform(subject)
        return subject
