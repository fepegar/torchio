import torch
from ....torchio import DATA
from ....utils import is_image_dict
from .. import RandomTransform


class RandomFlip(RandomTransform):
    def __init__(
            self,
            axes=(0,),
            flip_probability=0.5,
            seed=None,
            verbose=False,
            ):
        super().__init__(seed=seed, verbose=verbose)
        self.axes = axes
        assert flip_probability > 0
        assert flip_probability <= 1
        self.flip_probability = flip_probability

    def apply_transform(self, sample):
        axes_to_flip_hot = self.get_params(self.axes, self.flip_probability)
        sample['random_flip'] = axes_to_flip_hot
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            tensor = image_dict[DATA]
            dims = []
            for dim, flip_this in enumerate(axes_to_flip_hot):
                if not flip_this:
                    continue
                actual_dim = dim + 1  # images are 4D
                dims.append(actual_dim)
            tensor = torch.flip(tensor, dims=dims)
            image_dict[DATA] = tensor
        return sample

    @staticmethod
    def get_params(axes, probability):
        axes_hot = [False, False, False]
        for axis in axes:
            random_number = torch.rand(1)
            flip_this = bool(probability > random_number)
            axes_hot[axis] = flip_this
        return axes_hot
