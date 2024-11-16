from ...transform import TypeMaskingMethod
from .label_transform import LabelTransform


class RemapLabels(LabelTransform):
    r"""Modify labels in a label map.

    Masking can be used to split the label into two during
    the `inverse transformation <invertibility>`_.

    Args:
        remapping: Dictionary that specifies how labels should be remapped.
            The keys are the old labels, and the corresponding values replace
            them.
        masking_method: Defines a mask for where the label remapping is applied. It can be one of:

            - ``None``: the mask image is all ones, i.e. all values in the image are used.

            - A string: key to a :class:`torchio.LabelMap` in the subject which is used as a mask,
              OR an anatomical label: ``'Left'``, ``'Right'``, ``'Anterior'``, ``'Posterior'``,
              ``'Inferior'``, ``'Superior'`` which specifies a half of the mask volume to be ones.

            - A function: the mask image is computed as a function of the intensity image.
              The function must receive and return a 4D :class:`torch.Tensor`.

        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. plot::

        import torchio as tio

        subject = tio.datasets.FPG()
        subject.remove_image('t1')

        background_labels = (0, 1, 2, 3, 4)

        csf_labels = (5, 12, 16, 47, 52, 53)

        white_matter_labels = (
            45, 46,
            66, 67,
            81, 82,
            83, 84,
            85, 86,
            87,
            89, 90,
            91, 92,
            93, 94,
        )

        not_gray_matter_labels = (
            background_labels
            + csf_labels
            + white_matter_labels
        )

        gray_matter_labels = [
            label for label in subject.GIF_COLORS
            if label not in not_gray_matter_labels
        ]

        labels_groups = (
            background_labels,
            gray_matter_labels,
            white_matter_labels,
            csf_labels,
        )
        remapping = {}
        for target, labels in enumerate(labels_groups):
            for label in labels:
                remapping[label] = target

        parcellation_to_tissues = tio.RemapLabels(remapping)
        tissues = parcellation_to_tissues(subject).seg
        subject.add_image(tissues, 'remapped')
        subject.plot()

    Example:

        >>> import torch
        >>> import torchio as tio
        >>> def get_image(*labels):
        ...     tensor = torch.as_tensor(labels).reshape(1, 1, 1, -1)
        ...     image = tio.LabelMap(tensor=tensor)
        ...     return image
        ...
        >>> image = get_image(0, 1, 2, 3, 4)
        >>> remapping = {1: 2, 2: 1, 3: 1, 4: 7}
        >>> transform = tio.RemapLabels(remapping)
        >>> transform(image).data
        tensor([[[[0, 2, 1, 1, 7]]]])

    .. warning::
        The transform will not be correctly inverted if one of the values in
        ``remapping`` is also in the input image::

            >>> tensor = torch.as_tensor([0, 1]).reshape(1, 1, 1, -1)
            >>> subject = tio.Subject(label=tio.LabelMap(tensor=tensor))
            >>> mapping = {3: 1}  # the value 1 is in the input image
            >>> transform = tio.RemapLabels(mapping)
            >>> transformed = transform(subject)
            >>> back = transformed.apply_inverse_transform()
            >>> original_label_set = set(subject.label.data.unique().tolist())
            >>> back_label_set = set(back.label.data.unique().tolist())
            >>> original_label_set
            {0, 1}
            >>> back_label_set
            {0, 3}

    Example:

        >>> import torchio as tio
        >>> # Target label map has the following labels:
        >>> # {
        >>> #     'left_ventricle': 1, 'right_ventricle': 2,
        >>> #     'left_caudate': 3,   'right_caudate': 4,
        >>> #     'left_putamen': 5,   'right_putamen': 6,
        >>> #     'left_thalamus': 7,  'right_thalamus': 8,
        >>> # }
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
    """

    def __init__(
        self,
        remapping: dict[int, int],
        masking_method: TypeMaskingMethod = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.remapping = remapping
        self.masking_method = masking_method
        self.args_names = ['remapping', 'masking_method']

    def apply_transform(self, subject):
        for image in self.get_images(subject):
            original_label_set = set(image.data.unique().tolist())
            source_label_set = set(self.remapping.keys())
            # Do nothing if no keys in the mapping are found in the image
            if not source_label_set.intersection(original_label_set):
                continue
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
        # Not always, as explained in the docstring
        return True

    def inverse(self):
        targets = self.remapping.values()
        unique_targets = set(targets)
        if len(unique_targets) < len(targets):
            message = (
                'Labels mapping cannot be inverted because original values'
                f' are not unique: {self.remapping}'
            )
            raise RuntimeError(message)
        inverse_remapping = {v: k for k, v in self.remapping.items()}
        inverse_transform = RemapLabels(
            inverse_remapping,
            masking_method=self.masking_method,
            **self.kwargs,
        )
        return inverse_transform
