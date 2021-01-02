from collections import defaultdict
from typing import Union, Tuple, Dict, List

import torch
import numpy as np

from ....typing import TypeData
from ....data.subject import Subject
from ... import IntensityTransform
from .. import RandomTransform


class RandomBiasField(RandomTransform, IntensityTransform):
    r"""Add random MRI bias field artifact.

    MRI magnetic field inhomogeneity creates intensity
    variations of very low frequency across the whole image.

    The bias field is modeled as a linear combination of
    polynomial basis functions, as in K. Van Leemput et al., 1999,
    *Automated model-based tissue classification of MR images of the brain*.

    It was implemented in NiftyNet by Carole Sudre and used in
    `Sudre et al., 2017, Longitudinal segmentation of age-related
    white matter hyperintensities
    <https://www.sciencedirect.com/science/article/pii/S1361841517300257?via%3Dihub>`_.

    Args:
        coefficients: Maximum magnitude :math:`n` of polynomial coefficients.
            If a tuple :math:`(a, b)` is specified, then
            :math:`n \sim \mathcal{U}(a, b)`.
        order: Order of the basis polynomial functions.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            coefficients: Union[float, Tuple[float, float]] = 0.5,
            order: int = 3,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.coefficients_range = self._parse_range(
            coefficients, 'coefficients_range')
        self.order = _parse_order(order)

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        for image_name in self.get_images_dict(subject):
            coefficients = self.get_params(
                self.order,
                self.coefficients_range,
            )
            arguments['coefficients'][image_name] = coefficients
            arguments['order'][image_name] = self.order
        transform = BiasField(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(
            self,
            order: int,
            coefficients_range: Tuple[float, float],
            ) -> List[float]:
        # Sampling of the appropriate number of coefficients for the creation
        # of the bias field map
        random_coefficients = []
        for x_order in range(0, order + 1):
            for y_order in range(0, order + 1 - x_order):
                for _ in range(0, order + 1 - (x_order + y_order)):
                    number = self.sample_uniform(*coefficients_range)
                    random_coefficients.append(number.item())
        return random_coefficients


class BiasField(IntensityTransform):
    r"""Add MRI bias field artifact.

    Args:
        coefficients: Magnitudes of the polinomial coefficients.
        order: Order of the basis polynomial functions.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            coefficients: Union[List[float], Dict[str, List[float]]],
            order: Union[int, Dict[str, int]],
            **kwargs
            ):
        super().__init__(**kwargs)
        self.coefficients = coefficients
        self.order = order
        self.invert_transform = False
        self.args_names = 'coefficients', 'order'

    def arguments_are_dict(self):
        coefficients_dict = isinstance(self.coefficients, dict)
        order_dict = isinstance(self.order, dict)
        if coefficients_dict != order_dict:
            message = 'If one of the arguments is a dict, all must be'
            raise ValueError(message)
        return coefficients_dict and order_dict

    def apply_transform(self, subject: Subject) -> Subject:
        coefficients, order = self.coefficients, self.order
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                coefficients, order = self.coefficients[name], self.order[name]
            bias_field = self.generate_bias_field(
                image.data, order, coefficients)
            if self.invert_transform:
                np.divide(1, bias_field, out=bias_field)
            image.set_data(image.data * torch.from_numpy(bias_field))
        return subject

    @staticmethod
    def generate_bias_field(
            data: TypeData,
            order: int,
            coefficients: TypeData,
            ) -> np.ndarray:
        # Create the bias field map using a linear combination of polynomial
        # functions and the coefficients previously sampled
        shape = np.array(data.shape[1:])  # first axis is channels
        half_shape = shape / 2

        ranges = [np.arange(-n, n) + 0.5 for n in half_shape]

        bias_field = np.zeros(shape)
        meshes = np.asarray(np.meshgrid(*ranges))

        for mesh in meshes:
            mesh_max = mesh.max()
            if mesh_max > 0:
                mesh /= mesh_max
        x_mesh, y_mesh, z_mesh = meshes

        i = 0
        for x_order in range(order + 1):
            for y_order in range(order + 1 - x_order):
                for z_order in range(order + 1 - (x_order + y_order)):
                    coefficient = coefficients[i]
                    new_map = (
                        coefficient
                        * x_mesh ** x_order
                        * y_mesh ** y_order
                        * z_mesh ** z_order
                    )
                    bias_field += np.transpose(new_map, (1, 0, 2))  # why?
                    i += 1
        bias_field = np.exp(bias_field).astype(np.float32)
        return bias_field


def _parse_order(order):
    if not isinstance(order, int):
        raise TypeError(f'Order must be an int, not {type(order)}')
    if order < 0:
        raise ValueError(f'Order must be a positive int, not {order}')
    return order
