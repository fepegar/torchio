from collections.abc import Sequence

from ...transform import TypeMaskingMethod
from .remap_labels import RemapLabels


class RemoveLabels(RemapLabels):
    r"""Remove labels from a label map.

    The removed labels are remapped to the background label.

    This transformation is not `invertible <invertibility>`_.

    Args:
        labels: A sequence of label integers that will be removed.
        background_label: integer that specifies which label is considered to
            be background (typically, ``0``).
        masking_method: See :class:`~torchio.transforms.RemapLabels`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. plot::

        import torchio as tio
        colin = tio.datasets.Colin27(2008)
        label_map = colin.cls
        colin.remove_image('t2')
        colin.remove_image('pd')
        names_to_remove = (
            'Fat',
            'Muscles',
            'Skin and Muscles',
            'Skull',
            'Fat 2',
            'Dura',
            'Marrow'
        )
        labels = [colin.NAME_TO_LABEL[name] for name in names_to_remove]
        skull_stripping = tio.RemoveLabels(labels)
        only_brain = skull_stripping(label_map)
        colin.add_image(only_brain, 'brain')
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
        colin.plot(cmap_dict={'cls': colors, 'brain': colors})
    """

    def __init__(
        self,
        labels: Sequence[int],
        background_label: int = 0,
        masking_method: TypeMaskingMethod = None,
        **kwargs,
    ):
        remapping = {label: background_label for label in labels}
        super().__init__(remapping, masking_method, **kwargs)
        self.labels = labels
        self.background_label = background_label
        self.masking_method = masking_method
        self.args_names = ['labels', 'background_label', 'masking_method']

    def is_invertible(self):
        return False
