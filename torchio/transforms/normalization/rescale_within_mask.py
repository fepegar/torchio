import torch
import numpy as np
from ..torchio import INTENSITY
from ..utils import is_image_dict
from .transform import Transform
import numpy.ma as ma


class RescaleMask(Transform):
    def __init__(
            self,
            mask_field_name,
            out_min_max=(0, 1),
            percentiles=(1, 99),
            verbose=False,
            ):
        super().__init__(verbose=verbose)
        self.mask_field_name = mask_field_name
        self.out_min, self.out_max = out_min_max
        self.percentiles = percentiles

    def apply_transform(self, sample):
        """
        This could probably be written in two or three lines
        """
        mask_data = sample[self.mask_field_name]['data']
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] != INTENSITY:
                continue
            array = image_dict['data'].numpy()
            array_mask = ma.masked_array(array, np.logical_not(mask_data)).compressed()

            pa, pb = self.percentiles
            cutoff = np.percentile(array_mask, (pa, pb))
            np.clip(array, *cutoff, out=array)
            array -= array.min()  # [0, max]
            array /= array.max()  # [0, 1]
            out_range = self.out_max - self.out_min
            array *= out_range  # [0, out_range]
            array -= self.out_min  # [out_min, out_max]
            image_dict['data'] = torch.from_numpy(array)
            return sample
