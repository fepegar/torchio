from .transform import Transform


class IntensityTransform(Transform):
    """Transform that modifies voxel intensities only."""
    @staticmethod
    def get_images(sample):
        return sample.get_images(intensity_only=True)

    @staticmethod
    def get_images_dict(sample):
        return sample.get_images_dict(intensity_only=True)
