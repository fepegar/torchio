"""
The bias field is modelled as a linear combination of
polynomial basis functions, as in

    K. Van Leemput et al., 1999
    Automated model-based tissue classification of MR images of the brain


It was included in NiftyNet by Carole Sudre and used in:

    C. Sudre et al., 2017
    Longitudinal segmentation of age-related white matter hyperintensities
"""

import numpy as np
import torch
from ....torchio import INTENSITY, DATA
from ....utils import is_image_dict
from .. import RandomTransform


class RandomBiasField(RandomTransform):
    def __init__(
            self,
            coefficients_range=(-0.5, 0.5),
            order=3,
            proportion_to_augment=0.5,
            seed=None,
            verbose=False,
            ):
        super().__init__(seed=seed, verbose=verbose)
        self.coefficients_range = coefficients_range
        self.order = order
        self.proportion_to_augment = self.parse_probability(
            proportion_to_augment,
            'proportion_to_augment',
        )

    def apply_transform(self, sample):
        for image_name, image_dict in sample.items():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] != INTENSITY:
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
    def get_params(order, coefficients_range, probability):
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
    def generate_bias_field(data, order, coefficients):
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
