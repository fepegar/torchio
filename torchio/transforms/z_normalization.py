from .transform import Transform


class ZNormalization(Transform):
    def apply_transform(self, sample):
        znorm(sample['image'])
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
