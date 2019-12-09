"""
Adapted from NiftyNet
"""

import torch
import numpy as np


class ZNormalization:
    def __init__(self, verbose=False):
        """
        Assume single channel
        """
        self.verbose = verbose

    def __call__(self, sample):
        if self.verbose:
            import time
            start = time.time()
        znorm(sample['image'])
        if self.verbose:
            duration = time.time() - start
            print(f'ZNormalization: {duration:.1f} seconds')
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
