from __future__ import annotations

from collections import defaultdict
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np
import torch

from .. import RandomTransform
from ... import IntensityTransform
from ....data.subject import Subject
from ....typing import TypeTripletInt
from ....typing import TypeTuple
from ....utils import to_tuple


TypeLocations = Sequence[Tuple[TypeTripletInt, TypeTripletInt]]
TensorArray = TypeVar('TensorArray', np.ndarray, torch.Tensor)


class RandomSwap(RandomTransform, IntensityTransform):
    r"""Randomly swap patches within an image.

    This is typically used in `context restoration for self-supervised learning
    <https://www.sciencedirect.com/science/article/pii/S1361841518304699>`_.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to swap patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.
        num_iterations: Number of times that two patches will be swapped.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(
        self,
        patch_size: TypeTuple = 15,
        num_iterations: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = np.array(to_tuple(patch_size))
        self.num_iterations = self._parse_num_iterations(num_iterations)

    @staticmethod
    def _parse_num_iterations(num_iterations):
        if not isinstance(num_iterations, int):
            raise TypeError(
                f'num_iterations must be an int,not {num_iterations}',
            )
        if num_iterations < 0:
            raise ValueError(
                f'num_iterations must be positive,not {num_iterations}',
            )
        return num_iterations

    @staticmethod
    def get_params(
        tensor: torch.Tensor,
        patch_size: np.ndarray,
        num_iterations: int,
    ) -> List[Tuple[TypeTripletInt, TypeTripletInt]]:
        si, sj, sk = tensor.shape[-3:]
        spatial_shape = si, sj, sk  # for mypy
        locations = []
        for _ in range(num_iterations):
            first_ini, first_fin = get_random_indices_from_shape(
                spatial_shape,
                patch_size.tolist(),
            )
            while True:
                second_ini, second_fin = get_random_indices_from_shape(
                    spatial_shape,
                    patch_size.tolist(),
                )
                larger_than_initial = np.all(second_ini >= first_ini)
                less_than_final = np.all(second_fin <= first_fin)
                if larger_than_initial and less_than_final:
                    continue  # patches overlap
                else:
                    break  # patches don't overlap
            location = tuple(first_ini), tuple(second_ini)
            locations.append(location)
        return locations  # type: ignore[return-value]

    def apply_transform(self, subject: Subject) -> Subject:
        arguments: Dict[str, dict] = defaultdict(dict)
        for name, image in self.get_images_dict(subject).items():
            locations = self.get_params(
                image.data,
                self.patch_size,
                self.num_iterations,
            )
            arguments['locations'][name] = locations
            arguments['patch_size'][name] = self.patch_size
        transform = Swap(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed


class Swap(IntensityTransform):
    r"""Swap patches within an image.

    This is typically used in `context restoration for self-supervised learning
    <https://www.sciencedirect.com/science/article/pii/S1361841518304699>`_.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to swap patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.
        num_iterations: Number of times that two patches will be swapped.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(
        self,
        patch_size: Union[TypeTripletInt, Dict[str, TypeTripletInt]],
        locations: Union[TypeLocations, Dict[str, TypeLocations]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.locations = locations
        self.patch_size = patch_size
        self.args_names = ['locations', 'patch_size']
        self.invert_transform = False

    def apply_transform(self, subject: Subject) -> Subject:
        locations, patch_size = self.locations, self.patch_size
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                assert isinstance(self.locations, dict)
                assert isinstance(self.patch_size, dict)
                locations = self.locations[name]
                patch_size = self.patch_size[name]
            if self.invert_transform:
                assert isinstance(locations, list)
                locations.reverse()
            swapped = _swap(image.data, patch_size, locations)  # type: ignore[arg-type]  # noqa: B950
            image.set_data(swapped)
        return subject


def _swap(
    tensor: torch.Tensor,
    patch_size: TypeTuple,
    locations: List[Tuple[np.ndarray, np.ndarray]],
) -> torch.Tensor:
    # Note this function modifies the input in-place
    tensor = tensor.clone()
    patch_size_array = np.array(patch_size)
    for first_ini, second_ini in locations:
        first_fin = first_ini + patch_size_array
        second_fin = second_ini + patch_size_array
        first_patch = _crop(tensor, first_ini, first_fin)
        second_patch = _crop(tensor, second_ini, second_fin).clone()
        _insert(tensor, first_patch, second_ini)
        _insert(tensor, second_patch, first_ini)
    return tensor


def _insert(
    tensor: TensorArray,
    patch: TensorArray,
    index_ini: np.ndarray,
) -> None:
    index_fin = index_ini + np.array(patch.shape[-3:])
    i_ini, j_ini, k_ini = index_ini
    i_fin, j_fin, k_fin = index_fin
    tensor[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = patch


def _crop(
    image: TensorArray,
    index_ini: np.ndarray,
    index_fin: np.ndarray,
) -> TensorArray:
    i_ini, j_ini, k_ini = index_ini
    i_fin, j_fin, k_fin = index_fin
    return image[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]


def get_random_indices_from_shape(
    spatial_shape: Sequence[int],
    patch_size: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    assert len(spatial_shape) == 3
    assert len(patch_size) in (1, 3)
    shape_array = np.array(spatial_shape)
    patch_size_array = np.array(patch_size)
    max_index_ini_unchecked = shape_array - patch_size_array
    if (max_index_ini_unchecked < 0).any():
        message = (
            f'Patch size {patch_size} cannot be'
            f' larger than image spatial shape {spatial_shape}'
        )
        raise ValueError(message)
    max_index_ini = max_index_ini_unchecked.astype(np.uint16)
    coordinates = []
    for max_coordinate in max_index_ini.tolist():
        if max_coordinate == 0:
            coordinate = 0
        else:
            coordinate = int(torch.randint(max_coordinate, size=(1,)).item())
        coordinates.append(coordinate)
    index_ini = np.array(coordinates, np.uint16)
    index_fin = index_ini + patch_size_array
    return index_ini, index_fin
