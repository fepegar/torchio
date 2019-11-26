import torch
import numpy as np


class RandomFlip:
    def __init__(self, axes, flip_probability=0.5, verbose=False):
        self.axes = axes
        assert flip_probability > 0
        assert flip_probability <= 1
        self.flip_probability = flip_probability
        self.verbose = verbose

    def __call__(self, sample):
        """
        https://github.com/facebookresearch/InferSent/issues/99#issuecomment-446175325
        """
        if self.verbose:
            import time
            start = time.time()
        axes_to_flip_hot = self.get_params(self.axes, self.flip_probability)
        sample['random_flip'] = axes_to_flip_hot
        for key in 'image', 'label', 'sampler':
            if key not in sample:
                continue
            array = sample[key]
            for axis, flip_this in enumerate(axes_to_flip_hot):
                if not flip_this:
                    continue
                actual_axis = axis + 1  # images are 4D
                array = np.flip(array, axis=actual_axis).copy()
                sample[key] = array
        if self.verbose:
            duration = time.time() - start
            print(f'RandomFlip: {duration:.1f} seconds')
        return sample

    @staticmethod
    def get_params(axes, probability):
        axes_hot = [False, False, False]
        for axis in axes:
            random_number = torch.rand(1)
            flip_this = bool(probability > random_number)
            axes_hot[axis] = flip_this
        return axes_hot
