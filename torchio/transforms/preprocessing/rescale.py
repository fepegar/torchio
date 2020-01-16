import torch
import numpy as np
from ...torchio import DATA
from .normalization_transform import NormalizationTransform


class Rescale(NormalizationTransform):
    def __init__(
            self,
            out_min_max=(-1, 1),
            percentiles=(1, 99),
            masking_method=None,
            verbose=False,
            ):
        super().__init__(masking_method=masking_method, verbose=verbose)
        self.out_min, self.out_max = out_min_max
        self.percentiles = percentiles

    def apply_normalization(self, sample, image_name, mask):
        """
        This could probably be written in two or three lines
        """
        image_dict = sample[image_name]
        image_dict[DATA] = self.rescale(image_dict[DATA], mask)

    def rescale(self, data, mask):
        array = data.numpy()
        mask = mask.numpy()
        values = array[mask]
        pa, pb = self.percentiles
        cutoff = np.percentile(values, (pa, pb))
        np.clip(array, *cutoff, out=array)
        array -= array.min()  # [0, max]
        array /= array.max()  # [0, 1]
        out_range = self.out_max - self.out_min
        array *= out_range  # [0, out_range]
        array -= self.out_min  # [out_min, out_max]
        return torch.from_numpy(array)
