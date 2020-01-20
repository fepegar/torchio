import warnings
import torch
import numpy as np
from ....torchio import DATA
from .normalization_transform import NormalizationTransform


class Rescale(NormalizationTransform):
    def __init__(
            self,
            out_min_max,
            percentiles=(0, 100),
            masking_method=None,
            verbose=False,
            ):
        super().__init__(masking_method=masking_method, verbose=verbose)
        self.out_min, self.out_max = out_min_max
        self.percentiles = percentiles

    def apply_normalization(self, sample, image_name, mask):
        image_dict = sample[image_name]
        image_dict[DATA] = self.rescale(image_dict[DATA], mask, image_name)

    def rescale(self, data, mask, image_name):
        array = data.numpy()
        mask = mask.numpy()
        values = array[mask]
        cutoff = np.percentile(values, self.percentiles)
        np.clip(array, *cutoff, out=array)
        array -= array.min()  # [0, max]
        array_max = array.max()
        if array_max == 0:
            message = (
                f'Rescaling image "{image_name}" not possible'
                ' due to division by zero'
            )
            warnings.warn(message)
            return data
        array /= array.max()  # [0, 1]
        out_range = self.out_max - self.out_min
        array *= out_range  # [0, out_range]
        array += self.out_min  # [out_min, out_max]
        return torch.from_numpy(array)
