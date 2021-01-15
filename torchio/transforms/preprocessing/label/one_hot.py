import torch.nn.functional as F  # noqa: N812

from .label_transform import LabelTransform


class OneHot(LabelTransform):
    r"""Reencode label maps using one-hot encoding.

    Args:
        num_classes: See :func:`~torch.nn.functional.one_hot`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            num_classes: int = -1,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.args_names = []

    def apply_transform(self, subject):
        for image in self.get_images(subject):
            assert image.data.ndim == 4 and image.data.shape[0] == 1
            data = image.data.squeeze()
            num_classes = -1 if self.num_classes is None else self.num_classes
            one_hot = F.one_hot(data.long(), num_classes=num_classes)
            image.set_data(one_hot.permute(3, 0, 1, 2).type(data.type()))
        return subject
