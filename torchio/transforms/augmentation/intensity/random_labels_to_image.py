
from typing import Tuple, Optional, Sequence, List
import torch
from ....torchio import DATA, AFFINE, TypeData, TypeRangeFloat
from ....utils import check_sequence
from ....data.subject import Subject
from ....data.image import ScalarImage
from ... import IntensityTransform
from .. import RandomTransform


class RandomLabelsToImage(RandomTransform, IntensityTransform):
    r"""Generate an image from a segmentation.

    Based on the works by Billot et al.: `A Learning Strategy for
    Contrast-agnostic MRI Segmentation <https://arxiv.org/abs/2003.01995>`_
    and `Partial Volume Segmentation of Brain MRI Scans of any Resolution and
    Contrast <https://arxiv.org/abs/2004.10221>`.

    Args:
        label_key: String designating the label map in the sample
            that will be used to generate the new image.
        used_labels: Sequence of integers designating the labels used
            to generate the new image. If categorical encoding is used,
            :py:attr:`label_channels` refers to the values of the
            categorical encoding. If one hot encoding or partial-volume
            label maps are used, :py:attr:`label_channels` refers to the
            channels of the label maps.
            Default uses all labels. Missing voxels will be filled with zero
            or with voxels from an already existing volume,
            see :py:attr:`image_key`.
        image_key: String designating the key to which the new volume will be
            saved. If this key corresponds to an already existing volume,
            missing voxels will be filled with the corresponding values
            in the original volume.
        mean: Sequence of means for each label.
            For each value :math:`v`, if a tuple :math:`(a, b)` is
            provided then :math:`v \sim \mathcal{U}(a, b)`.
            If None, py:attr:`default_mean` range will be used for every label.
            If not None and py:attr:`label_channels` is not None,
            py:attr:`mean` and py:attr:`label_channels` must have the
            same length.
        std: Sequence of standard deviations for each label.
            For each value :math:`v`, if a tuple :math:`(a, b)` is
            provided then :math:`v \sim \mathcal{U}(a, b)`.
            If None, py:attr:`default_std` range will be used for every label.
            If not None and py:attr:`label_channels` is not None,
            py:attr:`std` and py:attr:`label_channels` must have the
            same length.
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
        >>> from torchio import RandomLabelsToImage, RescaleIntensity, RandomBlur, Compose
        >>> from torchio.datasets import ICBM2009CNonlinearSymmetryc
        >>> sample = ICBM2009CNonlinearSymmetryc()
        >>> # Using the default parameters
        >>> transform = RandomLabelsToImage(label_key='tissues')
        >>> # Using custom mean and std
        >>> transform = RandomLabelsToImage(
        ...     label_key='tissues', mean=[0.33, 0.66, 1.], std=[0, 0, 0]
        ... )
        >>> # Discretizing the partial volume maps and blurring the result
        >>> simulation_transform = RandomLabelsToImage(
        ...     label_key='tissues', mean=[0.33, 0.66, 1.], std=[0, 0, 0], discretize=True
        ... )
        >>> blurring_transform = RandomBlur(std=0.3)
        >>> transform = Compose([simulation_transform, blurring_transform])
        >>> transformed = transform(sample)  # sample has a new key 'image_from_labels' with the simulated image
        >>> # Filling holes of the simulated image with the original T1 image
        >>> rescale_transform = RescaleIntensity((0, 1), (1, 99))   # Rescale intensity before filling holes
        >>> simulation_transform = RandomLabelsToImage(
        ...     label_key='tissues',
        ...     image_key='t1',
        ...     used_labels=[0, 1]
        ... )
        >>> transform = Compose([rescale_transform, simulation_transform])
        >>> transformed = transform(sample)  # sample's key 't1' has been replaced with the simulated image
    """
    def __init__(
            self,
            label_key: str,
            used_labels: Optional[Sequence[int]] = None,
            image_key: str = 'image_from_labels',
            mean: Optional[Sequence[TypeRangeFloat]] = None,
            std: Optional[Sequence[TypeRangeFloat]] = None,
            default_mean: TypeRangeFloat = (0.1, 0.9),
            default_std: TypeRangeFloat = (0.01, 0.1),
            discretize: bool = False,
            p: float = 1,
            seed: Optional[int] = None,
            keys: Optional[List[str]] = None,
            ):
        super().__init__(p=p, seed=seed, keys=keys)
        self.label_key = self.parse_label_key(label_key)
        self.used_labels = self.parse_used_labels(used_labels)
        self.mean, self.std = self.parse_mean_and_std(mean, std)
        self.default_mean = self.parse_gaussian_param(
            default_mean, 'default_mean')
        self.default_std = self.parse_gaussian_param(default_std, 'default_std')
        self.image_key = image_key
        self.discretize = discretize

    @staticmethod
    def parse_label_key(label_key: str) -> str:
        if not isinstance(label_key, str):
            message = f'"label_key" must be a string, not {type(label_key)}'
            raise TypeError(message)
        return label_key

    @staticmethod
    def parse_used_labels(used_labels: Sequence[int]) -> Sequence[int]:
        if used_labels is None:
            return None
        check_sequence(used_labels, 'used_labels')
        for e in used_labels:
            if not isinstance(e, int):
                message = f'"used_labels" elements must be integers, ' \
                          f'not {used_labels}'
                raise ValueError(message)
        return used_labels

    def parse_mean_and_std(
            self,
            mean: Sequence[TypeRangeFloat],
            std: Sequence[TypeRangeFloat]
            ) -> (List[TypeRangeFloat], List[TypeRangeFloat]):
        if mean is not None:
            mean = self.parse_gaussian_params(mean, 'mean')
        if std is not None:
            std = self.parse_gaussian_params(std, 'std')
        if mean is not None and std is not None:
            message = (
                'If both "mean" and "std" are defined they must have the same'
                'length'
            )
            assert len(mean) == len(std), message
        return mean, std

    def parse_gaussian_params(
            self,
            params: Sequence[TypeRangeFloat],
            name: str
            ) -> List[TypeRangeFloat]:
        check_sequence(params, name)
        params = [
            self.parse_gaussian_param(p, f'{name}[{i}]')
            for i, p in enumerate(params)
        ]
        if self.used_labels is not None:
            message = (
                f'If both "{name}" and "used_labels" are defined, '
                f'they must have the same length'
            )
            assert len(params) == len(self.used_labels), message
        return params

    @staticmethod
    def parse_gaussian_param(
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

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {'mean': [], 'std': []}
        original_image = sample.get(self.image_key)

        label_map = sample[self.label_key][DATA]
        affine = sample[self.label_key][AFFINE]

        spatial_shape = label_map.shape[1:]

        # Find out if we face a partial-volume or a label map.
        # One hot encoded label map is considered as a partial-volume one.
        is_discretized = label_map.eq(label_map.round()).all() and \
            label_map.squeeze().dim() < label_map.dim()

        if not is_discretized and self.discretize:
            # Take label with highest value in voxel
            max_label, label_map = label_map.max(dim=0, keepdim=True)
            # Remove values where all labels are 0 (i.e. missing labels)
            label_map[max_label == 0] = -1
            is_discretized = True

        tissues = torch.zeros(1, *spatial_shape).float()
        if is_discretized:
            labels = label_map.unique().long().tolist()
            if -1 in labels:
                labels.remove(-1)
        else:
            labels = range(label_map.shape[0])

        # Raise error if mean and std are not defined for every label
        self.check_mean_and_std_length(labels)

        for label in labels:
            if self.used_labels is None or label in self.used_labels:
                mean, std = self.get_params(label)
                if is_discretized:
                    mask = label_map == label
                else:
                    mask = label_map[label]
                tissues += self.generate_tissue(mask, mean, std)

                random_parameters_images_dict['mean'].append(mean)
                random_parameters_images_dict['std'].append(std)
            else:
                # Modify label map to easily compute background mask
                if is_discretized:
                    label_map[label_map == label] = -1
                else:
                    label_map[label] = 0

        final_image = ScalarImage(affine=affine, tensor=tissues)

        if original_image is not None:
            if is_discretized:
                bg_mask = label_map == -1
            else:
                bg_mask = label_map.sum(dim=0, keepdim=True) < 0.5
            final_image[DATA][bg_mask] = original_image[DATA][bg_mask]

        sample.add_image(final_image, self.image_key)
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    def check_mean_and_std_length(self, labels: Sequence):
        if self.mean is not None:
            message = (
                '"mean" must define a value for each label but length of "mean"'
                f' is {len(self.mean)} while {len(labels)} labels were found'
            )
            assert len(self.mean) == len(labels), message
        if self.std is not None:
            message = (
                '"std" must define a value for each label but length of "std"'
                f' is {len(self.std)} while {len(labels)} labels were found'
            )
            assert len(self.std) == len(labels), message

    def get_params(
            self,
            label: int
            ) -> Tuple[float, float]:
        if self.mean is not None:
            mean_range = self.mean[label]
        else:
            mean_range = self.default_mean
        if self.std is not None:
            std_range = self.std[label]
        else:
            std_range = self.default_std

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
