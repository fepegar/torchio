from collections import defaultdict
from typing import Tuple, Union, List, Sequence, Dict

import torch
import numpy as np

from ....data.subject import Subject
from ....utils import to_tuple
from ....typing import TypeTuple, TypeData, TypeTripletInt
from ... import IntensityTransform
from .. import RandomTransform


TypeLocations = Sequence[Tuple[TypeTripletInt, TypeTripletInt]]


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
            raise TypeError('num_iterations must be an int,'
                            f'not {num_iterations}')
        if num_iterations < 0:
            raise ValueError('num_iterations must be positive,'
                             f'not {num_iterations}')
        return num_iterations

    @staticmethod
    def get_params(
            tensor: torch.Tensor,
            patch_size: np.ndarray,
            num_iterations: int,
            ) -> List[Tuple[TypeTripletInt, TypeTripletInt]]:
        spatial_shape = tensor.shape[-3:]
        locations = []
        for _ in range(num_iterations):
            first_ini, first_fin = get_random_indices_from_shape(
                spatial_shape,
                patch_size,
            )
            while True:
                second_ini, second_fin = get_random_indices_from_shape(
                    spatial_shape,
                    patch_size,
                )
                larger_than_initial = np.all(second_ini >= first_ini)
                less_than_final = np.all(second_fin <= first_fin)
                if larger_than_initial and less_than_final:
                    continue  # patches overlap
                else:
                    break  # patches don't overlap
            location = tuple(first_ini), tuple(second_ini)
            locations.append(location)
        return locations

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
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
            **kwargs
            ):
        super().__init__(**kwargs)
        self.locations = locations
        self.patch_size = patch_size
        self.args_names = 'locations', 'patch_size'
        self.invert_transform = False

    def apply_transform(self, subject: Subject) -> Subject:
        locations, patch_size = self.locations, self.patch_size
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                locations = self.locations[name]
                patch_size = self.patch_size[name]
            if self.invert_transform:
                locations.reverse()
            image.set_data(swap(image.data, patch_size, locations))
        return subject


def swap(
        tensor: torch.Tensor,
        patch_size: TypeTuple,
        locations: List[Tuple[np.ndarray, np.ndarray]],
        ) -> None:
    tensor = tensor.clone()
    patch_size = np.array(patch_size)
    for first_ini, second_ini in locations:
        first_fin = first_ini + patch_size
        second_fin = second_ini + patch_size
        first_patch = crop(tensor, first_ini, first_fin)
        second_patch = crop(tensor, second_ini, second_fin).clone()
        insert(tensor, first_patch, second_ini)
        insert(tensor, second_patch, first_ini)
    return tensor


def insert(tensor: TypeData, patch: TypeData, index_ini: np.ndarray) -> None:
    index_fin = index_ini + np.array(patch.shape[-3:])
    i_ini, j_ini, k_ini = index_ini
    i_fin, j_fin, k_fin = index_fin
    tensor[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = patch


def crop(
        image: Union[np.ndarray, torch.Tensor],
        index_ini: np.ndarray,
        index_fin: np.ndarray,
        ) -> Union[np.ndarray, torch.Tensor]:
    i_ini, j_ini, k_ini = index_ini
    i_fin, j_fin, k_fin = index_fin
    return image[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]


def get_random_indices_from_shape(
        spatial_shape: TypeTripletInt,
        patch_size: TypeTripletInt,
        ) -> Tuple[np.ndarray, np.ndarray]:
    shape_array = np.array(spatial_shape)
    patch_size_array = np.array(patch_size)
    max_index_ini = shape_array - patch_size_array
    if (max_index_ini < 0).any():
        message = (
            f'Patch size {patch_size} cannot be'
            f' larger than image spatial shape {spatial_shape}'
        )
        raise ValueError(message)
    max_index_ini = max_index_ini.astype(np.uint16)
    coordinates = []
    for max_coordinate in max_index_ini.tolist():
        if max_coordinate == 0:
            coordinate = 0
        else:
            coordinate = torch.randint(max_coordinate, size=(1,)).item()
        coordinates.append(coordinate)
    index_ini = np.array(coordinates, np.uint16)
    index_fin = index_ini + patch_size_array
    return index_ini, index_fin
