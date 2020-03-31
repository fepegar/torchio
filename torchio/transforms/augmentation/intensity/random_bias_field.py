
from typing import Union, Tuple, Optional
import numpy as np
import torch
from ....torchio import INTENSITY, DATA, TYPE, TypeData
from ....utils import is_image_dict
from .. import RandomTransform


class RandomBiasField(RandomTransform):
    r"""Add random MRI bias field artifact.

    Args:
        coefficients: Magnitude :math:`n` of polynomial coefficients.
            If a tuple :math:`(a, b)` is specified, then
            :math:`n \sim \mathcal{U}(a, b)`.
        order: Order of the basis polynomial functions.
        proportion_to_augment: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
    """
    def __init__(
            self,
            coefficients: Union[float, Tuple[float, float]] = 0.5,
            order: int = 3,
            proportion_to_augment: float = 1,
            seed: Optional[int] = None,
            ):
        super().__init__(seed=seed)
        self.coefficients_range = self.parse_range(
            coefficients, 'coefficients_range')
        self.order = order
        self.proportion_to_augment = self.parse_probability(
            proportion_to_augment,
            'proportion_to_augment',
        )

    def apply_transform(self, sample: dict) -> dict:
        for image_name, image_dict in sample.items():
            if not is_image_dict(image_dict):
                continue
            if image_dict[TYPE] != INTENSITY:
                continue
            do_augmentation, coefficients = self.get_params(
                self.order,
                self.coefficients_range,
                self.proportion_to_augment,
            )
            sample[image_name]['random_bias_coefficients'] = coefficients
            sample[image_name]['random_bias_do_augmentation'] = do_augmentation
            if not do_augmentation:
                continue
            bias_field = self.generate_bias_field(
                image_dict[DATA], self.order, coefficients)
            image_with_bias = image_dict[DATA] * torch.from_numpy(bias_field)
            image_dict[DATA] = image_with_bias
        return sample

    @staticmethod
    def get_params(
            order: int,
            coefficients_range: Tuple[float, float],
            probability: float,
            ) -> Tuple[bool, np.ndarray]:
        """
        Sampling of the appropriate number of coefficients for the creation
        of the bias field map
        """
        random_coefficients = []
        for x_order in range(0, order + 1):
            for y_order in range(0, order + 1 - x_order):
                for z_order in range(0, order + 1 - (x_order + y_order)):
                    number = torch.FloatTensor(1).uniform_(*coefficients_range)
                    random_coefficients.append(number.item())
        do_augmentation = torch.rand(1) < probability
        return do_augmentation, np.array(random_coefficients)

    @staticmethod
    def generate_bias_field(
            data: TypeData,
            order: int,
            coefficients: TypeData,
            ) -> np.ndarray:
        """
        Create the bias field map using a linear combination of polynomial
        functions and the coefficients previously sampled
        """
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
