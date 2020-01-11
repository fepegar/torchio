import torch
from .. import Transform
from ...torchio import INTENSITY
from ...utils import is_image_dict


class NormalizationTransform(Transform):
    def __init__(self, masking_method=None, verbose=False):
        super().__init__(verbose=verbose)
        self.masking_method = masking_method
        if self.masking_method is None:
            self.masking_method = self.ones
        if isinstance(self.masking_method, str):
            self.mask_name = self.masking_method
        else:
            self.mask_name = None

    def get_mask(self, sample, data):
        if self.mask_name is None:
            return self.masking_method(data)
        else:
            return sample[self.mask_name]['data'].bool()

    def apply_transform(self, sample):
        for image_name, image_dict in sample.items():
            if not is_image_dict(image_dict):
                continue
            if not image_dict['type'] == INTENSITY:
                continue
            mask = self.get_mask(sample, image_dict['data'])
            self.apply_normalization(sample, image_name, mask)
        return sample

    def apply_normalization(self, sample, image_name, mask):
        """There must be a nicer way of doing this"""
        raise NotImplementedError

    @staticmethod
    def ones(data):
        return torch.ones_like(data, dtype=torch.bool)

    @staticmethod
    def mean(data):
        mask = data > data.mean()
        return mask
