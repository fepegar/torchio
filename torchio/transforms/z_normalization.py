from ..utils import is_image_dict
from .transform import Transform


class ZNormalization(Transform):
    def apply_transform(self, sample):
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            znorm(image_dict['data'])
        return sample


def znorm(data, masking_function=None):
    if masking_function is None:
        masking_function = mean_plus
    mask_data = masking_function(data)
    values = data[mask_data]
    mean, std = values.mean(), values.std()
    data -= mean
    data /= std


def mean_plus(data):
    return data > data.mean()
