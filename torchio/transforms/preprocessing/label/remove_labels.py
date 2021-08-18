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

    .. plot::

        import torchio as tio
        subject = tio.datasets.Colin27(2008)
        label_map = subject.cls
        subject.remove_image('t2')
        subject.remove_image('pd')
        remove_labels = tio.RemoveLabels([4, 5, 6, 7, 9, 10, 11, 12])
        only_brain = remove_labels(label_map)
        subject.add_image(only_brain, 'brain')
        colors = {
            0: (0, 0, 0),
            1: (127, 255, 212),
            2: (96, 204, 96),
            3: (240, 230, 140),
            4: (176, 48, 96),
            5: (48, 176, 96),
            6: (220, 247, 164),
            7: (103, 255, 255),
            9: (205, 62, 78),
            10: (238, 186, 243),
            11: (119, 159, 176),
            12: (220, 216, 20),
        }
        subject.plot(cmap_dict={'cls': colors, 'brain': colors})
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
