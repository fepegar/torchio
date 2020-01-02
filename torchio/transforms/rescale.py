import torch
import numpy as np
from ..torchio import INTENSITY
from ..utils import is_image_dict
from .transform import Transform


class Rescale(Transform):
    def __init__(
            self,
            out_min_max=(-1, 1),
            percentiles=(1, 99),
            verbose=False,
            ):
        super().__init__(verbose=verbose)
        self.out_min, self.out_max = out_min_max
        self.percentiles = percentiles

    def apply_transform(self, sample):
        """
        This could probably be written in two or three lines
        """
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] != INTENSITY:
                continue
            array = image_dict['data'].numpy()
            pa, pb = self.percentiles
            cutoff = np.percentile(array, (pa, pb))
            np.clip(array, *cutoff, out=array)
            array -= array.min()  # [0, max]
            array /= array.max()  # [0, 1]
            out_range = self.out_max - self.out_min
            array *= out_range  # [0, out_range]
            array -= self.out_min  # [out_min, out_max]
            image_dict['data'] = torch.from_numpy(array)
            return sample
