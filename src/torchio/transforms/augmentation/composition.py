from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Union

import numpy as np
import torch

from ...data.subject import Subject
from ..transform import Transform
from . import RandomTransform

TypeTransformsDict = Union[dict[Transform, float], Sequence[Transform]]


class Compose(Transform):
    """Compose several transforms together.

    Args:
        transforms: Sequence of instances of
            :class:`~torchio.transforms.Transform`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, transforms: Sequence[Transform], **kwargs):
        super().__init__(parse_input=False, **kwargs)
        for transform in transforms:
            if not callable(transform):
                message = (
                    'One or more of the objects passed to the Compose'
                    f' transform are not callable: "{transform}"'
                )
                raise TypeError(message)
        self.transforms = list(transforms)

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, index) -> Transform:
        return self.transforms[index]

    def __repr__(self) -> str:
        return f'{self.name}({self.transforms})'

    def get_base_args(self) -> dict:
        init_args = super().get_base_args()
        if 'parse_input' in init_args:
            init_args.pop('parse_input')
        return init_args

    def apply_transform(self, subject: Subject) -> Subject:
        for transform in self.transforms:
            subject = transform(subject)  # type: ignore[assignment]
        return subject

    def is_invertible(self) -> bool:
        return all(t.is_invertible() for t in self.transforms)

    def inverse(self, warn: bool = True) -> Compose:
        """Return a composed transform with inverted order and transforms.

        Args:
            warn: Issue a warning if some transforms are not invertible.
        """
        transforms = []
        for transform in self.transforms:
            if transform.is_invertible():
                transforms.append(transform.inverse())
            elif warn:
                message = f'Skipping {transform.name} as it is not invertible'
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        transforms.reverse()
        result = Compose(transforms, **self.get_base_args())
        if not transforms and warn:
            warnings.warn(
                'No invertible transforms found',
                RuntimeWarning,
                stacklevel=2,
            )
        return result


class OneOf(RandomTransform):
    """Apply only one of the given transforms.

    Args:
        transforms: Dictionary with instances of
            :class:`~torchio.transforms.Transform` as keys and
            probabilities as values. Probabilities are normalized so they sum
            to one. If a sequence is given, the same probability will be
            assigned to each transform.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> colin = tio.datasets.Colin27()
        >>> transforms_dict = {
        ...     tio.RandomAffine(): 0.75,
        ...     tio.RandomElasticDeformation(): 0.25,
        ... }  # Using 3 and 1 as probabilities would have the same effect
        >>> transform = tio.OneOf(transforms_dict)
        >>> transformed = transform(colin)
    """

    def __init__(self, transforms: TypeTransformsDict, **kwargs):
        super().__init__(parse_input=False, **kwargs)
        self.transforms_dict = self._get_transforms_dict(transforms)

    def get_base_args(self) -> dict:
        init_args = super().get_base_args()
        if 'parse_input' in init_args:
            init_args.pop('parse_input')
        return init_args

    def apply_transform(self, subject: Subject) -> Subject:
        weights = torch.Tensor(list(self.transforms_dict.values()))
        index = torch.multinomial(weights, 1)
        transforms = list(self.transforms_dict.keys())
        transform = transforms[index]
        transformed = transform(subject)
        return transformed  # type: ignore[return-value]

    def _get_transforms_dict(
        self,
        transforms: TypeTransformsDict,
    ) -> dict[Transform, float]:
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
    def _normalize_probabilities(
        transforms_dict: dict[Transform, float],
    ) -> None:
        probabilities = np.array(list(transforms_dict.values()), dtype=float)
        if np.any(probabilities < 0):
            message = (
                f'Probabilities must be greater or equal to zero, not "{probabilities}"'
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
