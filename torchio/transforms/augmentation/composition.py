from typing import Union, Sequence

import torch
import numpy as np
from torchvision.transforms import Compose as PyTorchCompose

from ...data.subject import Subject
from .. import Transform
from . import RandomTransform


class Compose(Transform):
    """Compose several transforms together.

    Args:
        transforms: Sequence of instances of
            :py:class:`~torchio.transforms.transform.Transform`.
        p: Probability that this transform will be applied.

    .. note::
        This is a thin wrapper of :py:class:`torchvision.transforms.Compose`.
    """
    def __init__(self, transforms: Sequence[Transform], p: float = 1):
        super().__init__(p=p)
        self.transform = PyTorchCompose(transforms)

    def apply_transform(self, sample: Subject):
        return self.transform(sample)


class OneOf(RandomTransform):
    """Apply only one of the given transforms.

    Args:
        transforms: Dictionary with instances of
            :py:class:`~torchio.transforms.transform.Transform` as keys and
            probabilities as values. Probabilities are normalized so they sum
            to one. If a sequence is given, the same probability will be
            assigned to each transform.
        p: Probability that this transform will be applied.

    Example:
        >>> import torchio
        >>> ixi = torchio.datasets.ixi.IXITiny('ixi', download=True)
        >>> sample = ixi[0]
        >>> transforms_dict = {
        ...     torchio.transforms.RandomAffine(): 0.75,
        ...     torchio.transforms.RandomElasticDeformation(): 0.25,
        ... }  # Using 3 and 1 as probabilities would have the same effect
        >>> transform = torchio.transforms.OneOf(transforms_dict)

    """
    def __init__(
            self,
            transforms: Union[dict, Sequence[Transform]],
            p: float = 1,
            ):
        super().__init__(p=p)
        self.transforms_dict = self._get_transforms_dict(transforms)

    def apply_transform(self, sample: Subject):
        weights = torch.Tensor(list(self.transforms_dict.values()))
        index = torch.multinomial(weights, 1)
        transforms = list(self.transforms_dict.keys())
        transform = transforms[index]
        transformed = transform(sample)
        return transformed

    def _get_transforms_dict(self, transforms: Union[dict, Sequence]):
        if isinstance(transforms, dict):
            transforms_dict = dict(transforms)
            self._normalize_probabilities(transforms_dict)
        else:
            try:
                p = 1 / len(transforms)
            except TypeError as e:
                message = (
                    'Transforms argument must be a dictionary or a sequence,'
                    f' not {type(transforms)}'
                )
                raise ValueError(message) from e
            transforms_dict = {transform: p for transform in transforms}
        for transform in transforms_dict:
            if not isinstance(transform, Transform):
                message = (
                    'All keys in transform_dict must be instances of'
                    f'torchio.Transform, not "{type(transform)}"'
                )
                raise ValueError(message)
        return transforms_dict

    @staticmethod
    def _normalize_probabilities(transforms_dict: dict):
        probabilities = np.array(list(transforms_dict.values()), dtype=float)
        if np.any(probabilities < 0):
            message = (
                'Probabilities must be greater or equal to zero,'
                f' not "{probabilities}"'
            )
            raise ValueError(message)
        if np.all(probabilities == 0):
            message = (
                'At least one probability must be greater than zero,'
                f' but they are "{probabilities}"'
            )
            raise ValueError(message)
        for transform, probability in transforms_dict.items():
            transforms_dict[transform] = probability / probabilities.sum()
