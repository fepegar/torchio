
from typing import Union, Tuple, Optional, Dict, Sequence
import torch
import numpy as np
from ....torchio import DATA, TypeData, TypeRangeFloat, TypeNumber, AFFINE, INTENSITY
from ....data.subject import Subject
from ....data.image import Image
from .. import RandomTransform

MEAN_RANGE = (0.1, 0.9)
STD_RANGE = (0.01, 0.1)


class RandomLabelsToImage(RandomTransform):
    r"""Generate a random image from a binary label map or a set of
     partial volume (PV) label maps.

    Args:
        label_key: String designating the label map in the sample
            that will be used to generate the new image.
            Cannot be set at the same time as pv_label_keys.
        pv_label_keys: Sequence of strings designating the PV label maps in
            the sample that will be used to generate the new image.
            Cannot be set at the same time as label_key.
        image_key: String designating the key to which the new volume will be saved.
            If this key corresponds to an already existing volume, zero elements from
            the label maps will be filled with elements of the original volume.
        gaussian_parameters: Dictionary containing the mean and standard deviation for
            each label. For each value :math:`v`, if a tuple
            :math:`(a, b)` is provided then
            :math:`v \sim \mathcal{U}(a, b)`.
            If no value is given for a label, value from default_gaussian_parameters
            will be used.
        default_gaussian_parameters: Dictionary containing the default
            mean and standard deviation used for all labels that are not
            defined in gaussian_parameters.
            Default values are :math:`(0.1, 0.9)` for the mean and
            :math:`(0.01, 0.1)` for the standard deviation.
        binarize: Boolean to tell if PV label maps should be binarized.
            Does not have any effects if not using PV label maps.
            Binarization is done taking the highest value per voxel
            in the different PV label maps.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.

    .. note:: Generated images using label maps are unrealistic, therefore it is recommended
        to blur the new images. See :py:class:`~torchio.transforms.augmentation.RandomBlur`.

    Example:
        >>> import torchio
        >>> from torchio import RandomLabelsToImage, DATA, RescaleIntensity, Compose
        >>> from torchio.datasets import Colin27
        >>> colin = Colin27(2008)
        >>> # Using the default gaussian_parameters
        >>> transform = RandomLabelsToImage(label_key='cls')
        >>> # Using custom gaussian_parameters
        >>> label_values = colin['cls'][DATA].unique().round()
        >>> gaussian_parameters = {
        >>>     int(label): {
        >>>         'mean': i / len(label_values), 'std': 0.01
        >>>     } for i, label in enumerate(label_values)
        >>> }
        >>> transform = RandomLabelsToImage(label_key='cls', gaussian_parameters=gaussian_parameters)
        >>> transformed = transform(colin)  # colin has a new key 'image' with the simulated image
        >>> # Filling holes of the simulated image with the original T1 image
        >>> rescale_transform = RescaleIntensity((0, 1), (1, 99))   # Rescale intensity before filling holes
        >>> simulation_transform = RandomLabelsToImage(
        >>>     label_key='cls',
        >>>     image_key='t1',
        >>>     gaussian_parameters={0: {'mean': 0, 'std': 0}}
        >>> )
        >>> transform = Compose([rescale_transform, simulation_transform])
        >>> transformed = transform(colin)  # colin's key 't1' has been replaced with the simulated image
    """
    def __init__(
            self,
            label_key: Optional[str] = None,
            pv_label_keys: Optional[Sequence[str]] = None,
            image_key: str = 'image',
            gaussian_parameters: Optional[Dict[Union[str, TypeNumber], Dict[str, TypeRangeFloat]]] = None,
            default_gaussian_parameters: Optional[Dict[str, TypeRangeFloat]] = None,
            binarize: bool = False,
            p: float = 1,
            seed: Optional[int] = None,
            ):
        super().__init__(p=p, seed=seed)
        self.label_key, self.pv_label_keys = self.parse_keys(label_key, pv_label_keys)
        self.default_gaussian_parameters = self.parse_default_gaussian_parameters(default_gaussian_parameters)
        self.gaussian_parameters = self.parse_gaussian_parameters(gaussian_parameters)
        self.image_key = image_key
        self.binarize = binarize

    @staticmethod
    def parse_keys(label_key, pv_label_keys):
        if label_key is not None and pv_label_keys is not None:
            raise ValueError('"label_key" and "pv_label_keys" can\'t be set at the same time.')
        if label_key is None and pv_label_keys is None:
            raise ValueError('One of "label_key" and "pv_label_keys" must be set.')
        if label_key is not None and not isinstance(label_key, str):
            raise TypeError(f'"label_key" must be a string, not {label_key}')
        if pv_label_keys is not None:
            try:
                iter(pv_label_keys)
            except TypeError:
                raise TypeError(f'"pv_label_keys" must be a sequence of strings, not {pv_label_keys}')
            for key in pv_label_keys:
                if not isinstance(key, str):
                    raise TypeError(f'Every key of "pv_label_keys" must be a string, found {key}')
            pv_label_keys = list(pv_label_keys)

        return label_key, pv_label_keys

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        original_image = sample.get(self.image_key)

        if self.pv_label_keys is not None:
            label_map, affine = self.parse_pv_label_maps(self.pv_label_keys, sample)
            n_labels, *image_shape = label_map.shape
            labels = self.pv_label_keys
            values = list(range(n_labels))

            if self.binarize:
                # Take label with highest value in voxel
                max_label, label_map = label_map.max(dim=0)
                # Remove values where all labels are 0
                label_map[max_label == 0] = -1

        else:
            label_map = sample[self.label_key][DATA][0]
            affine = sample[self.label_key][AFFINE]
            image_shape = label_map.shape
            values = label_map.unique()
            labels = [int(key) for key in values.round()]

        tissues = torch.zeros(image_shape)

        for i, label in enumerate(labels):
            mean, std = self.get_params(label)
            if self.pv_label_keys is not None and not self.binarize:
                mask = label_map[i]
            else:
                mask = label_map == values[i]
            tissues += self.generate_tissue(mask, mean, std)

            random_parameters_images_dict[label] = {
                'mean': mean,
                'std': std
            }

        final_image = Image(type=INTENSITY, affine=affine, tensor=tissues)

        if original_image is not None:
            if self.pv_label_keys is not None and not self.binarize:
                label_map = label_map.sum(dim=0)
            background_indices = label_map.unsqueeze(0) <= 0
            final_image[DATA][background_indices] = original_image[DATA][background_indices]

        sample[self.image_key] = final_image
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    def parse_default_gaussian_parameters(self, default_gaussian_parameters):
        if default_gaussian_parameters is None:
            return {'mean': MEAN_RANGE, 'std': STD_RANGE}

        if list(default_gaussian_parameters.keys()) != ['mean', 'std']:
            raise KeyError(f'Default gaussian parameters {default_gaussian_parameters.keys()} do not '
                           f'match {["mean", "std"]}')

        mean = self.parse_gaussian_parameter(default_gaussian_parameters['mean'], 'mean')
        std = self.parse_gaussian_parameter(default_gaussian_parameters['std'], 'std')

        return {'mean': mean, 'std': std}

    def parse_gaussian_parameters(self, gaussian_parameters):
        if gaussian_parameters is None:
            gaussian_parameters = {}

        if self.pv_label_keys is not None:
            if not set(self.pv_label_keys).issuperset(gaussian_parameters.keys()):
                raise KeyError(f'Found keys in gaussian parameters {gaussian_parameters.keys()} '
                               f'not in pv_label_keys {self.pv_label_keys}')

        parsed_gaussian_parameters = {}

        for label_key, dictionary in gaussian_parameters.items():
            if 'mean' in dictionary:
                mean = self.parse_gaussian_parameter(dictionary['mean'], 'mean')
            else:
                mean = self.default_gaussian_parameters['mean']
            if 'std' in dictionary:
                std = self.parse_gaussian_parameter(dictionary['std'], 'std')
            else:
                std = self.default_gaussian_parameters['std']
            parsed_gaussian_parameters.update({label_key: {'mean': mean, 'std': std}})

        return parsed_gaussian_parameters

    @staticmethod
    def parse_gaussian_parameter(
            nums_range: TypeRangeFloat,
            name: str,
            ) -> Tuple[float, float]:
        if isinstance(nums_range, (int, float)):
            return nums_range, nums_range

        if len(nums_range) != 2:
            raise ValueError(
                f'If {name} is a sequence,'
                f' it must be of len 2, not {nums_range}')
        min_value, max_value = nums_range
        if min_value > max_value:
            raise ValueError(
                f'If {name} is a sequence, the second value must be'
                f' equal or greater than the first, not {nums_range}')
        return min_value, max_value

    @staticmethod
    def parse_pv_label_maps(
            pv_label_keys: Sequence[str],
            sample: dict,
            ) -> (TypeData, TypeData):
        try:
            label_map = torch.cat([sample[key][DATA] for key in pv_label_keys], dim=0)
        except RuntimeError:
            raise RuntimeError('PV label maps have different shapes, make sure they all have the same shapes.')
        affine = sample[pv_label_keys[0]][AFFINE]
        for key in pv_label_keys[1:]:
            if not np.array_equal(affine, sample[key][AFFINE]):
                raise RuntimeWarning('Be careful, PV label maps with different affines were found.')
        return label_map, affine

    def get_params(
            self,
            label: Union[str, TypeNumber]
            ) -> Tuple[float, float]:
        if label in self.gaussian_parameters:
            mean_range, std_range = self.gaussian_parameters[label]['mean'], self.gaussian_parameters[label]['std']
        else:
            mean_range, std_range = self.default_gaussian_parameters['mean'], self.default_gaussian_parameters['std']

        mean = torch.FloatTensor(1).uniform_(*mean_range).item()
        std = torch.FloatTensor(1).uniform_(*std_range).item()
        return mean, std

    @staticmethod
    def generate_tissue(
            data: TypeData,
            mean: TypeNumber,
            std: TypeNumber,
            ) -> TypeData:
        # Create the simulated tissue using a gaussian random variable
        data_shape = data.shape
        gaussian = torch.randn(data_shape) * std + mean
        return gaussian * data
