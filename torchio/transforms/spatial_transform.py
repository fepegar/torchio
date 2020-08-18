from .transform import Transform


class SpatialTransform(Transform):
    """Transform that modifies image bounds or voxels positions."""
    @staticmethod
    def get_images(sample):
        return sample.get_images(intensity_only=False)

    @staticmethod
    def get_images_dict(sample):
        return sample.get_images_dict(intensity_only=False)
