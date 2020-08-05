
from typing import Union, Tuple, Optional, Dict, Sequence, List
import torch
import numpy as np
from ....torchio import DATA, TypeData, TypeRangeFloat, TypeNumber, AFFINE
from ....data.subject import Subject
from ....data.image import ScalarImage
from .. import RandomTransform


TypeGaussian = Optional[Dict[Union[str, TypeNumber], Dict[str, TypeRangeFloat]]]


class RandomLabelsToImage(RandomTransform):
    r"""Generate an image from a segmentation.

    Based on the work by `Billot et al., A Learning Strategy for
    Contrast-agnostic MRI Segmentation <https://arxiv.org/abs/2003.01995>`_.

    Args:
        label_key: String designating the label map in the sample
            that will be used to generate the new image.
            Cannot be set at the same time as :py:attr:`pv_label_keys`.
        pv_label_keys: Sequence of strings designating the partial-volume
            label maps in the sample that will be used to generate the new
            image. Cannot be set at the same time as :py:attr:`label_key`.
        image_key: String designating the key to which the new volume will be
            saved. If this key corresponds to an already existing volume,
            voxels that have a value of 0 in the label maps will be filled with
            the corresponding values in the original volume.
        gaussian_parameters: Dictionary containing the mean and standard
            deviation for each label. For each value :math:`v`, if a tuple
            :math:`(a, b)` is provided then :math:`v \sim \mathcal{U}(a, b)`.
            If no value is given for a label, :py:attr:`default_mean` and
            :py:attr:`default_std` ranges will be used.
        default_mean: Default mean range.
        default_std: Default standard deviation range.
        discretize: If ``True``, partial-volume label maps will be discretized.
            Does not have any effects if not using partial-volume label maps.
            Discretization is done taking the class of the highest value per
            voxel in the different partial-volume label maps using
            :py:func:`torch.argmax()` on the channel dimension (i.e. 0).
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.
        keys: See :py:class:`~torchio.transforms.Transform`.

    .. note:: It is recommended to blur the new images to make the result more
        realistic. See
        :py:class:`~torchio.transforms.augmentation.RandomBlur`.

    Example:
        >>> import torchio
        >>> from torchio import RandomLabelsToImage, DATA, RescaleIntensity, Compose
        >>> from torchio.datasets import Colin27
        >>> colin = Colin27(2008)
        >>> # Using the default gaussian_parameters
        >>> transform = RandomLabelsToImage(label_key='cls')
        >>> # Using custom gaussian_parameters
        >>> label_values = colin['cls'][DATA].unique().round().long()
        >>> gaussian_parameters = {
        ...     label: {
        ...         'mean': i / len(label_values),
        ...         'std': 0.01
        ...     }
        ...     for i, label in enumerate(label_values)
        ... }
        >>> transform = RandomLabelsToImage(label_key='cls', gaussian_parameters=gaussian_parameters)
        >>> transformed = transform(colin)  # colin has a new key 'image' with the simulated image
        >>> # Filling holes of the simulated image with the original T1 image
        >>> rescale_transform = RescaleIntensity((0, 1), (1, 99))   # Rescale intensity before filling holes
        >>> simulation_transform = RandomLabelsToImage(
        ...     label_key='cls',
        ...     image_key='t1',
        ...     gaussian_parameters={0: {'mean': 0, 'std': 0}}
        ... )
        >>> transform = Compose([rescale_transform, simulation_transform])
        >>> transformed = transform(colin)  # colin's key 't1' has been replaced with the simulated image
    """
    def __init__(
            self,
            label_key: Optional[str] = None,
            pv_label_keys: Optional[Sequence[str]] = None,
            image_key: str = 'image',
            gaussian_parameters: TypeGaussian = None,
            default_mean: TypeRangeFloat = (0.1, 0.9),
            default_std: TypeRangeFloat = (0.01, 0.1),
            discretize: bool = False,
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.label_key, self.pv_label_keys = self.parse_keys(
            label_key, pv_label_keys)
        self.default_mean = self.parse_gaussian_parameter(default_mean, 'mean')
        self.default_std = self.parse_gaussian_parameter(default_std, 'std')
        self.gaussian_parameters = self.parse_gaussian_parameters(
            gaussian_parameters)
        self.image_key = image_key
        self.discretize = discretize

    @staticmethod
    def parse_keys(
            label_key: str,
            pv_label_keys: Sequence[str]
            ) -> (str, Sequence[str]):
        if label_key is not None and pv_label_keys is not None:
            message = (
                '"label_key" and "pv_label_keys" cannot be set at the same time'
            )
            raise ValueError(message)
        if label_key is None and pv_label_keys is None:
            message = 'One of "label_key" and "pv_label_keys" must be set'
            raise ValueError(message)
        if label_key is not None and not isinstance(label_key, str):
            message = f'"label_key" must be a string, not {type(label_key)}'
            raise TypeError(message)
        if pv_label_keys is not None:
            try:
                iter(pv_label_keys)
            except TypeError:
                message = (
                    '"pv_label_keys" must be a sequence of strings, '
                    f'not {pv_label_keys}'
                )
                raise TypeError(message)
            for key in pv_label_keys:
                if not isinstance(key, str):
                    message = (
                        f'Every key of "pv_label_keys" must be a string, '
                        f'found {type(key)}'
                    )
                    raise TypeError(message)
            pv_label_keys = list(pv_label_keys)

        return label_key, pv_label_keys

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        original_image = sample.get(self.image_key)

        if self.pv_label_keys is not None:
            label_map, affine = self.parse_pv_label_maps(
                self.pv_label_keys, sample)
            n_labels, *image_shape = label_map.shape
            labels = self.pv_label_keys
            values = list(range(n_labels))

            if self.discretize:
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
            if self.pv_label_keys is not None and not self.discretize:
                mask = label_map[i]
            else:
                mask = label_map == values[i]
            tissues += self.generate_tissue(mask, mean, std)

            random_parameters_images_dict[label] = {
                'mean': mean,
                'std': std
            }

        final_image = ScalarImage(affine=affine, tensor=tissues)

        if original_image is not None:
            if self.pv_label_keys is not None and not self.discretize:
                label_map = label_map.sum(dim=0)
            bg_mask = label_map.unsqueeze(0) <= 0
            final_image[DATA][bg_mask] = original_image[DATA][bg_mask]

        sample.add_image(final_image, self.image_key)
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    def parse_gaussian_parameters(
            self,
            parameters: TypeGaussian,
            ) -> TypeGaussian:
        if parameters is None:
            parameters = {}

        if self.pv_label_keys is not None:
            if not set(self.pv_label_keys).issuperset(parameters.keys()):
                message = (
                    f'Found keys in gaussian parameters {parameters.keys()} '
                    f'not in pv_label_keys {self.pv_label_keys}'
                )
                raise KeyError(message)

        parsed_parameters = {}

        for label_key, dictionary in parameters.items():
            if 'mean' in dictionary:
                mean = self.parse_gaussian_parameter(dictionary['mean'], 'mean')
            else:
                mean = self.default_mean
            if 'std' in dictionary:
                std = self.parse_gaussian_parameter(dictionary['std'], 'std')
            else:
                std = self.default_std
            parsed_parameters.update({label_key: {'mean': mean, 'std': std}})

        return parsed_parameters

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
            label_map = torch.cat([sample[key][DATA] for key in pv_label_keys])
        except RuntimeError:
            message = 'Partial-volume label maps have different shapes'
            raise RuntimeError(message)
        affine = sample[pv_label_keys[0]][AFFINE]
        for key in pv_label_keys[1:]:
            if not np.array_equal(affine, sample[key][AFFINE]):
                message = (
                    'Partial-volume label maps have different affine matrices'
                )
                raise RuntimeWarning(message)
        return label_map, affine

    def get_params(
            self,
            label: Union[str, TypeNumber]
            ) -> Tuple[float, float]:
        if label in self.gaussian_parameters:
            mean_range = self.gaussian_parameters[label]['mean']
            std_range = self.gaussian_parameters[label]['std']
        else:
            mean_range, std_range = self.default_mean, self.default_std

        mean = torch.FloatTensor(1).uniform_(*mean_range).item()
        std = torch.FloatTensor(1).uniform_(*std_range).item()
        return mean, std

    @staticmethod
    def generate_tissue(
            data: TypeData,
            mean: float,
            std: float,
            ) -> TypeData:
        # Create the simulated tissue using a gaussian random variable
        data_shape = data.shape
        gaussian = torch.randn(data_shape) * std + mean
        return gaussian * data
