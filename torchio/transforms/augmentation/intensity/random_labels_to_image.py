from typing import Tuple, Optional, Sequence, List

import torch

from ....utils import check_sequence
from ....data.subject import Subject
from ....typing import TypeData, TypeRangeFloat
from ....data.image import ScalarImage, LabelMap
from ... import IntensityTransform
from .. import RandomTransform


class RandomLabelsToImage(RandomTransform, IntensityTransform):
    r"""Randomly generate an image from a segmentation.

    Based on the works by Billot et al.: `A Learning Strategy for Contrast-agnostic MRI Segmentation`_
    and `Partial Volume Segmentation of Brain MRI Scans of any Resolution and Contrast <https://link.springer.com/chapter/10.1007/978-3-030-59728-3_18>`__.

    .. _A Learning Strategy for Contrast-agnostic MRI Segmentation: http://proceedings.mlr.press/v121/billot20a.html

    Args:
        label_key: String designating the label map in the subject
            that will be used to generate the new image.
        used_labels: Sequence of integers designating the labels used
            to generate the new image. If categorical encoding is used,
            :attr:`label_channels` refers to the values of the
            categorical encoding. If one hot encoding or partial-volume
            label maps are used, :attr:`label_channels` refers to the
            channels of the label maps.
            Default uses all labels. Missing voxels will be filled with zero
            or with voxels from an already existing volume,
            see :attr:`image_key`.
        image_key: String designating the key to which the new volume will be
            saved. If this key corresponds to an already existing volume,
            missing voxels will be filled with the corresponding values
            in the original volume.
        mean: Sequence of means for each label.
            For each value :math:`v`, if a tuple :math:`(a, b)` is
            provided then :math:`v \sim \mathcal{U}(a, b)`.
            If ``None``, :attr:`default_mean` range will be used for every
            label.
            If not ``None`` and :attr:`label_channels` is not ``None``,
            :attr:`mean` and :attr:`label_channels` must have the
            same length.
        std: Sequence of standard deviations for each label.
            For each value :math:`v`, if a tuple :math:`(a, b)` is
            provided then :math:`v \sim \mathcal{U}(a, b)`.
            If ``None``, :attr:`default_std` range will be used for every
            label.
            If not ``None`` and :attr:`label_channels` is not ``None``,
            :attr:`std` and :attr:`label_channels` must have the
            same length.
        default_mean: Default mean range.
        default_std: Default standard deviation range.
        discretize: If ``True``, partial-volume label maps will be discretized.
            Does not have any effects if not using partial-volume label maps.
            Discretization is done taking the class of the highest value per
            voxel in the different partial-volume label maps using
            :func:`torch.argmax()` on the channel dimension (i.e. 0).
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. tip:: It is recommended to blur the new images to make the result more
        realistic. See
        :class:`~torchio.transforms.augmentation.RandomBlur`.

    Example:
        >>> import torchio as tio
        >>> subject = tio.datasets.ICBM2009CNonlinearSymmetric()
        >>> # Using the default parameters
        >>> transform = tio.RandomLabelsToImage(label_key='tissues')
        >>> # Using custom mean and std
        >>> transform = tio.RandomLabelsToImage(
        ...     label_key='tissues', mean=[0.33, 0.66, 1.], std=[0, 0, 0]
        ... )
        >>> # Discretizing the partial volume maps and blurring the result
        >>> simulation_transform = tio.RandomLabelsToImage(
        ...     label_key='tissues', mean=[0.33, 0.66, 1.], std=[0, 0, 0], discretize=True
        ... )
        >>> blurring_transform = tio.RandomBlur(std=0.3)
        >>> transform = tio.Compose([simulation_transform, blurring_transform])
        >>> transformed = transform(subject)  # subject has a new key 'image_from_labels' with the simulated image
        >>> # Filling holes of the simulated image with the original T1 image
        >>> rescale_transform = tio.RescaleIntensity((0, 1), (1, 99))   # Rescale intensity before filling holes
        >>> simulation_transform = tio.RandomLabelsToImage(
        ...     label_key='tissues',
        ...     image_key='t1',
        ...     used_labels=[0, 1]
        ... )
        >>> transform = tio.Compose([rescale_transform, simulation_transform])
        >>> transformed = transform(subject)  # subject's key 't1' has been replaced with the simulated image
    """  # noqa: E501
    def __init__(
            self,
            label_key: Optional[str] = None,
            used_labels: Optional[Sequence[int]] = None,
            image_key: str = 'image_from_labels',
            mean: Optional[Sequence[TypeRangeFloat]] = None,
            std: Optional[Sequence[TypeRangeFloat]] = None,
            default_mean: TypeRangeFloat = (0.1, 0.9),
            default_std: TypeRangeFloat = (0.01, 0.1),
            discretize: bool = False,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.label_key = _parse_label_key(label_key)
        self.used_labels = _parse_used_labels(used_labels)
        self.mean, self.std = self.parse_mean_and_std(mean, std)
        self.default_mean = self.parse_gaussian_parameter(
            default_mean, 'default_mean')
        self.default_std = self.parse_gaussian_parameter(
            default_std,
            'default_std',
        )
        self.image_key = image_key
        self.discretize = discretize

    def parse_mean_and_std(
            self,
            mean: Sequence[TypeRangeFloat],
            std: Sequence[TypeRangeFloat]
            ) -> (List[TypeRangeFloat], List[TypeRangeFloat]):
        if mean is not None:
            mean = self.parse_gaussian_parameters(mean, 'mean')
        if std is not None:
            std = self.parse_gaussian_parameters(std, 'std')
        if mean is not None and std is not None:
            message = (
                'If both "mean" and "std" are defined they must have the same'
                'length'
            )
            assert len(mean) == len(std), message
        return mean, std

    def parse_gaussian_parameters(
            self,
            params: Sequence[TypeRangeFloat],
            name: str
            ) -> List[TypeRangeFloat]:
        check_sequence(params, name)
        params = [
            self.parse_gaussian_parameter(p, f'{name}[{i}]')
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

    def apply_transform(self, subject: Subject) -> Subject:
        if self.label_key is None:
            iterable = subject.get_images_dict(intensity_only=False).items()
            for name, image in iterable:
                if isinstance(image, LabelMap):
                    self.label_key = name
                    break
            else:
                message = f'No label maps found in subject: {subject}'
                raise RuntimeError(message)

        arguments = {
            'label_key': self.label_key,
            'mean': [],
            'std': [],
            'image_key': self.image_key,
            'used_labels': self.used_labels,
            'discretize': self.discretize,
        }

        label_map = subject[self.label_key].data

        # Find out if we face a partial-volume image or a label map.
        # One-hot-encoded label map is considered as a partial-volume image
        all_discrete = label_map.eq(label_map.float().round()).all()
        same_num_dims = label_map.squeeze().dim() < label_map.dim()
        is_discretized = all_discrete and same_num_dims

        if not is_discretized and self.discretize:
            # Take label with highest value in voxel
            max_label, label_map = label_map.max(dim=0, keepdim=True)
            # Remove values where all labels are 0 (i.e. missing labels)
            label_map[max_label == 0] = -1
            is_discretized = True

        if is_discretized:
            labels = label_map.unique().long().tolist()
            if -1 in labels:
                labels.remove(-1)
        else:
            labels = range(label_map.shape[0])

        # Raise error if mean and std are not defined for every label
        _check_mean_and_std_length(labels, self.mean, self.std)

        for label in labels:
            mean, std = self.get_params(label)
            arguments['mean'].append(mean)
            arguments['std'].append(std)

        transform = LabelsToImage(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(self, label: int) -> Tuple[float, float]:
        if self.mean is None:
            mean_range = self.default_mean
        else:
            mean_range = self.mean[label]
        if self.std is None:
            std_range = self.default_std
        else:
            std_range = self.std[label]
        mean = self.sample_uniform(*mean_range).item()
        std = self.sample_uniform(*std_range).item()
        return mean, std


class LabelsToImage(IntensityTransform):
    r"""Generate an image from a segmentation.

    Args:
        label_key: String designating the label map in the subject
            that will be used to generate the new image.
        used_labels: Sequence of integers designating the labels used
            to generate the new image. If categorical encoding is used,
            :attr:`label_channels` refers to the values of the
            categorical encoding. If one hot encoding or partial-volume
            label maps are used, :attr:`label_channels` refers to the
            channels of the label maps.
            Default uses all labels. Missing voxels will be filled with zero
            or with voxels from an already existing volume,
            see :attr:`image_key`.
        image_key: String designating the key to which the new volume will be
            saved. If this key corresponds to an already existing volume,
            missing voxels will be filled with the corresponding values
            in the original volume.
        mean: Sequence of means for each label.
            If not ``None`` and :attr:`label_channels` is not ``None``,
            :attr:`mean` and :attr:`label_channels` must have the
            same length.
        std: Sequence of standard deviations for each label.
            If not ``None`` and :attr:`label_channels` is not ``None``,
            :attr:`std` and :attr:`label_channels` must have the
            same length.
        discretize: If ``True``, partial-volume label maps will be discretized.
            Does not have any effects if not using partial-volume label maps.
            Discretization is done taking the class of the highest value per
            voxel in the different partial-volume label maps using
            :func:`torch.argmax()` on the channel dimension (i.e. 0).
        seed: Seed for the random number generator.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. note:: It is recommended to blur the new images to make the result more
        realistic. See
        :class:`~torchio.transforms.augmentation.RandomBlur`.
    """
    def __init__(
            self,
            label_key: str,
            mean: Optional[Sequence[float]],
            std: Optional[Sequence[float]],
            image_key: str = 'image_from_labels',
            used_labels: Optional[Sequence[int]] = None,
            discretize: bool = False,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.label_key = _parse_label_key(label_key)
        self.used_labels = _parse_used_labels(used_labels)
        self.mean, self.std = mean, std
        self.image_key = image_key
        self.discretize = discretize
        self.args_names = (
            'label_key',
            'mean',
            'std',
            'image_key',
            'used_labels',
            'discretize',
        )

    def apply_transform(self, subject: Subject) -> Subject:
        original_image = subject.get(self.image_key)

        label_map_image = subject[self.label_key]
        label_map = label_map_image.data
        affine = label_map_image.affine

        # Find out if we face a partial-volume image or a label map.
        # One-hot-encoded label map is considered as a partial-volume image
        all_discrete = label_map.eq(label_map.float().round()).all()
        same_num_dims = label_map.squeeze().dim() < label_map.dim()
        is_discretized = all_discrete and same_num_dims

        if not is_discretized and self.discretize:
            # Take label with highest value in voxel
            max_label, label_map = label_map.max(dim=0, keepdim=True)
            # Remove values where all labels are 0 (i.e. missing labels)
            label_map[max_label == 0] = -1
            is_discretized = True

        tissues = torch.zeros(1, *label_map_image.spatial_shape).float()
        if is_discretized:
            labels = label_map.unique().long().tolist()
            if -1 in labels:
                labels.remove(-1)
        else:
            labels = range(label_map.shape[0])

        # Raise error if mean and std are not defined for every label
        _check_mean_and_std_length(labels, self.mean, self.std)

        for i, label in enumerate(labels):
            if self.used_labels is None or label in self.used_labels:
                mean = self.mean[i]
                std = self.std[i]
                if is_discretized:
                    mask = label_map == label
                else:
                    mask = label_map[label]
                tissues += self.generate_tissue(mask, mean, std)

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
            final_image.data[bg_mask] = original_image.data[bg_mask].float()

        subject.add_image(final_image, self.image_key)
        return subject

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


def _parse_label_key(label_key: Optional[str]) -> Optional[str]:
    if label_key is not None and not isinstance(label_key, str):
        message = (
            f'"label_key" must be a string or None, not {type(label_key)}')
        raise TypeError(message)
    return label_key


def _parse_used_labels(used_labels: Sequence[int]) -> Sequence[int]:
    if used_labels is None:
        return None
    check_sequence(used_labels, 'used_labels')
    for e in used_labels:
        if not isinstance(e, int):
            message = (
                'Items in "used_labels" must be integers,'
                f' but some are not: {used_labels}'
            )
            raise ValueError(message)
    return used_labels


def _check_mean_and_std_length(
        labels: Sequence[int],
        means: Optional[Sequence[TypeRangeFloat]],
        stds: Optional[Sequence[TypeRangeFloat]],
        ) -> None:
    num_labels = len(labels)
    if means is not None:
        num_means = len(means)
        message = (
            '"mean" must define a value for each label but length of "mean"'
            f' is {num_means} while {num_labels} labels were found'
        )
        if num_means != num_labels:
            raise RuntimeError(message)
    if stds is not None:
        num_stds = len(stds)
        message = (
            '"std" must define a value for each label but length of "std"'
            f' is {num_stds} while {num_labels} labels were found'
        )
        if num_stds != num_labels:
            raise RuntimeError(message)
