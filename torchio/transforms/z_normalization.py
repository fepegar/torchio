import torch
from ..torchio import INTENSITY
from ..utils import is_image_dict
from .transform import Transform


class ZNormalization(Transform):
    """
    Subtract mean and divide by standard deviation
    """
    def __init__(self, use_mean_threshold=True, verbose=False):
        super().__init__(verbose=verbose)
        self.use_mean_threshold = use_mean_threshold

    def apply_transform(self, sample):
        masking_function = mean_plus if self.use_mean_threshold else None
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] != INTENSITY:
                continue
            image_dict['data'] = znorm(
                image_dict['data'],
                masking_function=masking_function,
            )
        return sample


def znorm(data, masking_function):
    if masking_function is None:
        mask_data = torch.ones_like(data, dtype=torch.bool)
    else:
        mask_data = masking_function(data)
    values = data[mask_data]
    mean, std = values.mean(), values.std()
    data = data - mean
    data = data / std
    return data


def mean_plus(data):
    return data > data.mean()
