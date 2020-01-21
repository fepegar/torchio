from ....torchio import DATA
from .normalization_transform import NormalizationTransform


class ZNormalization(NormalizationTransform):
    """
    Subtract mean and divide by standard deviation
    """
    def __init__(self, masking_method=None, verbose=False):
        super().__init__(masking_method=masking_method, verbose=verbose)

    def apply_normalization(self, sample, image_name, mask):
        image_dict = sample[image_name]
        image_dict[DATA] = self.znorm(
            image_dict[DATA],
            mask,
        )

    @staticmethod
    def znorm(data, mask):
        values = data[mask]
        mean, std = values.mean(), values.std()
        data = data - mean
        data = data / std
        return data
