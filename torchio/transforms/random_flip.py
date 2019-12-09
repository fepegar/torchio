import torch
import numpy as np
from .random_transform import RandomTransform


class RandomFlip(RandomTransform):
    def __init__(self, axes, flip_probability=0.5, seed=None, verbose=False):
        super().__init__(seed=seed, verbose=verbose)
        self.axes = axes
        assert flip_probability > 0
        assert flip_probability <= 1
        self.flip_probability = flip_probability

    def apply_transform(self, sample):
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
                # https://github.com/facebookresearch/InferSent/issues/99#issuecomment-446175325
                array = np.flip(array, axis=actual_axis).copy()
                sample[key] = array
        return sample

    @staticmethod
    def get_params(axes, probability):
        axes_hot = [False, False, False]
        for axis in axes:
            random_number = torch.rand(1)
            flip_this = bool(probability > random_number)
            axes_hot[axis] = flip_this
        return axes_hot
