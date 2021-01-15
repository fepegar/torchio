from typing import Dict

from ...transform import TypeMaskingMethod
from .label_transform import LabelTransform


class RemapLabels(LabelTransform):
    r"""Remap the integer ids of labels in a LabelMap.

    This transformation may not be invertible if two labels are combined by the
    remapping.
    A masking method can be used to correctly split the label into two during
    the `inverse transformation <invertibility>`_ (see example).

    Args:
        remapping: Dictionary that specifies how labels should be remapped.
            The keys are the old label ids, and the corresponding values replace
            them.
        masking_method: Defines a mask for where the label remapping is applied. It can be one of:

            - ``None``: the mask image is all ones, i.e. all values in the image are used.

            - A string: key to a :class:`torchio.LabelMap` in the subject which is used as a mask,
              OR an anatomical label: ``'Left'``, ``'Right'``, ``'Anterior'``, ``'Posterior'``,
              ``'Inferior'``, ``'Superior'`` which specifies a side of the mask volume to be ones.

            - A function: the mask image is computed as a function of the intensity image.
              The function must receive and return a :class:`torch.Tensor`.

        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> # Target label map has the following labels:
        >>> # {'left_ventricle': 1, 'right_ventricle': 2, 'left_caudate': 3, 'right_caudate': 4,
        >>> #  'left_putamen': 5, 'right_putamen': 6, 'left_thalamus': 7, 'right_thalamus': 8}
        >>> transform = tio.RemapLabels({2:1, 4:3, 6:5, 8:7})
        >>> # Merge right side labels with left side labels
        >>> transformed = transform(subject)
        >>> # Undesired behavior: The inverse transform will remap ALL left side labels to right side labels
        >>> # so the label map only has right side labels.
        >>> inverse_transformed = transformed.apply_inverse_transform()
        >>> # Here's the *right* way to do it with masking:
        >>> transform = tio.RemapLabels({2:1, 4:3, 6:5, 8:7}, masking_method="Right")
        >>> # Remap the labels on the right side only (no difference yet).
        >>> transformed = transform(subject)
        >>> # Apply the inverse on the right side only. The labels are correctly split into left/right.
        >>> inverse_transformed = transformed.apply_inverse_transform()
    """  # noqa: E501
    def __init__(
            self,
            remapping: Dict[int, int],
            masking_method: TypeMaskingMethod = None,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.remapping = remapping
        self.masking_method = masking_method
        self.args_names = ('remapping', 'masking_method',)

    def apply_transform(self, subject):
        for image in self.get_images(subject):
            new_data = image.data.clone()
            mask = self.get_mask_from_masking_method(
                self.masking_method,
                subject,
                new_data,
            )
            for old_id, new_id in self.remapping.items():
                new_data[mask & (image.data == old_id)] = new_id
            image.set_data(new_data)

        return subject

    def is_invertible(self):
        return True

    def inverse(self):
        inverse_remapping = {v: k for k, v in self.remapping.items()}
        inverse_transform = RemapLabels(
            inverse_remapping,
            masking_method=self.masking_method,
            **self.kwargs,
        )
        return inverse_transform
