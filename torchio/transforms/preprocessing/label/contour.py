import SimpleITK as sitk

from .label_transform import LabelTransform


class Contour(LabelTransform):
    r"""Keep only the borders of each connected component in a binary image.

    Args:
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args_names = []

    def apply_transform(self, subject):
        for image in self.get_images(subject):
            assert image.data.ndim == 4 and image.data.shape[0] == 1
            sitk_image = image.as_sitk()
            contour = sitk.BinaryContour(sitk_image)
            tensor, _ = self.sitk_to_nib(contour)
            image.set_data(tensor)
        return subject
