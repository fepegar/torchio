from typing import Sequence

from ...transform import TypeMaskingMethod
from .remap_labels import RemapLabels


class RemoveLabels(RemapLabels):
    r"""Remove labels from a label map by remapping them to the background label.

    This transformation is not `invertible <invertibility>`_.

    Args:
        labels: A sequence of label integers that will be removed.
        background_label: integer that specifies which label is considered to
            be background (generally 0).
        masking_method: See :class:`~torchio.transforms.RemapLabels`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            labels: Sequence[int],
            background_label: int = 0,
            masking_method: TypeMaskingMethod = None,
            **kwargs
            ):
        remapping = {label: background_label for label in labels}
        super().__init__(remapping, masking_method, **kwargs)
        self.labels = labels
        self.background_label = background_label
        self.masking_method = masking_method
        self.args_names = ('labels', 'background_label', 'masking_method',)

    def is_invertible(self):
        return False
