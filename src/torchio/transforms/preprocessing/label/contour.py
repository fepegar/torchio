import SimpleITK as sitk

from .label_transform import LabelTransform


class Contour(LabelTransform):
    r"""Keep only the borders of each connected component in a binary image.

    Args:
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def apply_transform(self, subject):
        for image in self.get_images(subject):
            if image.num_channels > 1:
                message = (
                    'The number of input channels must be 1,'
                    f' but it is {image.num_channels}'
                )
                raise RuntimeError(message)
            sitk_image = image.as_sitk()
            contour = sitk.BinaryContour(sitk_image)
            tensor, _ = self.sitk_to_nib(contour)
            image.set_data(tensor)
        return subject
