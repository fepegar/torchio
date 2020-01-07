from ..torchio import INTENSITY
from ..utils import is_image_dict
from .transform import Transform


class ZNormalization(Transform):
    """
    Subtract mean and divide by standard deviation
    """
    def apply_transform(self, sample):
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] != INTENSITY:
                continue
            image_dict['data'] = znorm(image_dict['data'])
        return sample


def znorm(data, masking_function=None):
    if masking_function is None:
        masking_function = mean_plus
    mask_data = masking_function(data)
    values = data[mask_data]
    mean, std = values.mean(), values.std()
    data = data - mean
    data = data / std
    return data


def mean_plus(data):
    return data > data.mean()
