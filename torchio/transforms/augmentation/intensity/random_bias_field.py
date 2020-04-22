
from typing import Union, Tuple, Optional
import numpy as np
import torch
from ....torchio import DATA, TypeData
from ....data.subject import Subject
from .. import RandomTransform


class RandomBiasField(RandomTransform):
    r"""Add random MRI bias field artifact.

    Args:
        coefficients: Magnitude :math:`n` of polynomial coefficients.
            If a tuple :math:`(a, b)` is specified, then
            :math:`n \sim \mathcal{U}(a, b)`.
        order: Order of the basis polynomial functions.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
    """
    def __init__(
            self,
            coefficients: Union[float, Tuple[float, float]] = 0.5,
            order: int = 3,
            p: float = 1,
            seed: Optional[int] = None,
            ):
        super().__init__(p=p, seed=seed)
        self.coefficients_range = self.parse_range(
            coefficients, 'coefficients_range')
        self.order = order

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        for image_name, image_dict in sample.get_images_dict().items():
            coefficients = self.get_params(
                self.order,
                self.coefficients_range,
            )
            random_parameters_dict = {'coefficients': coefficients}
            random_parameters_images_dict[image_name] = random_parameters_dict

            bias_field = self.generate_bias_field(
                image_dict[DATA], self.order, coefficients)
            image_with_bias = image_dict[DATA] * torch.from_numpy(bias_field)
            image_dict[DATA] = image_with_bias
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    @staticmethod
    def get_params(
            order: int,
            coefficients_range: Tuple[float, float],
            ) -> Tuple[bool, np.ndarray]:
        # Sampling of the appropriate number of coefficients for the creation
        # of the bias field map
        random_coefficients = []
        for x_order in range(0, order + 1):
            for y_order in range(0, order + 1 - x_order):
                for _ in range(0, order + 1 - (x_order + y_order)):
                    number = torch.FloatTensor(1).uniform_(*coefficients_range)
                    random_coefficients.append(number.item())
        return np.array(random_coefficients)

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

        ranges = [np.arange(-n, n) for n in half_shape]

        bias_field = np.zeros(shape)
        x_mesh, y_mesh, z_mesh = np.asarray(np.meshgrid(*ranges))

        x_mesh /= x_mesh.max()
        y_mesh /= y_mesh.max()
        z_mesh /= z_mesh.max()

        i = 0
        for x_order in range(order + 1):
            for y_order in range(order + 1 - x_order):
                for z_order in range(order + 1 - (x_order + y_order)):
                    random_coefficient = coefficients[i]
                    new_map = (
                        random_coefficient
                        * x_mesh ** x_order
                        * y_mesh ** y_order
                        * z_mesh ** z_order
                    )
                    bias_field += np.transpose(new_map, (1, 0, 2))  # why?
                    i += 1
        bias_field = np.exp(bias_field).astype(np.float32)
        return bias_field
