import warnings
from typing import Tuple, Optional, Union
import torch
import numpy as np
import SimpleITK as sitk
from ....torchio import DATA, AFFINE
from ....data.subject import Subject
from .. import RandomTransform


class RandomGhosting(RandomTransform):
    r"""Add random MRI ghosting artifact.

    Args:
        num_ghosts: Number of 'ghosts' :math:`n` in the image.
            If :py:attr:`num_ghosts` is a tuple :math:`(a, b)`, then
            :math:`n \sim \mathcal{U}(a, b) \cap \mathbb{N}`.
        axes: Axis along which the ghosts will be created. If
            :py:attr:`axes` is a tuple, the axis will be randomly chosen
            from the passed values.
        intensity: Number between 0 and 1 representing the artifact strength
            :math:`s`. If ``0``, the ghosts will not be visible. If a tuple
            :math:`(a, b)`, is provided then
            :math:`s \sim \mathcal{U}(a, b)`.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.

    .. note:: The execution time of this transform does not depend on the
        number of ghosts.
    """
    def __init__(
            self,
            num_ghosts: Union[int, Tuple[int, int]] = (4, 10),
            axes: Union[int, Tuple[int, ...]] = (0, 1, 2),
            intensity: Union[float, Tuple[float, float]] = (0.5, 1),
            p: float = 1,
            seed: Optional[int] = None,
            ):
        super().__init__(p=p, seed=seed)
        if not isinstance(axes, tuple):
            try:
                axes = tuple(axes)
            except TypeError:
                axes = (axes,)
        for axis in axes:
            if axis not in (0, 1, 2):
                raise ValueError(f'Axes must be in (0, 1, 2), not "{axes}"')
        self.axes = axes
        if isinstance(num_ghosts, int):
            self.num_ghosts_range = num_ghosts, num_ghosts
        elif isinstance(num_ghosts, tuple) and len(num_ghosts) == 2:
            self.num_ghosts_range = num_ghosts
        self.intensity_range = self.parse_range(intensity, 'intensity')
        for n in self.intensity_range:
            if not 0 <= n <= 1:
                message = (
                    f'Intensity must be a number between 0 and 1, not {n}')
                raise ValueError(message)

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        for image_name, image_dict in sample.get_images_dict().items():
            data = image_dict[DATA]
            is_2d = data.shape[-3] == 1
            axes = [a for a in self.axes if a != 0] if is_2d else self.axes
            params = self.get_params(
                self.num_ghosts_range,
                axes,
                self.intensity_range,
            )
            num_ghosts_param, axis_param, intensity_param = params
            random_parameters_dict = {
                'axis': axis_param,
                'num_ghosts': num_ghosts_param,
                'intensity': intensity_param,
            }
            random_parameters_images_dict[image_name] = random_parameters_dict
            if (data[0] < -0.1).any():
                # I use -0.1 instead of 0 because Python was warning me when
                # a value in a voxel was -7.191084e-35
                # There must be a better way of solving this
                message = (
                    f'Image "{image_name}" from "{image_dict["stem"]}"'
                    ' has negative values.'
                    ' Results can be unexpected because the transformed sample'
                    ' is computed as the absolute values'
                    ' of an inverse Fourier transform'
                )
                warnings.warn(message)
            image = self.nib_to_sitk(
                data[0],
                image_dict[AFFINE],
            )
            data = self.add_artifact(
                image,
                num_ghosts_param,
                axis_param,
                intensity_param,
            )
            # Add channels dimension
            data = data[np.newaxis, ...]
            image_dict[DATA] = torch.from_numpy(data)
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    @staticmethod
    def get_params(
            num_ghosts_range: Tuple[int, int],
            axes: Tuple[int, ...],
            intensity_range: Tuple[float, float],
            ) -> Tuple:
        ng_min, ng_max = num_ghosts_range
        num_ghosts = torch.randint(ng_min, ng_max + 1, (1,)).item()
        axis = axes[torch.randint(0, len(axes), (1,))]
        intensity = torch.FloatTensor(1).uniform_(*intensity_range).item()
        return num_ghosts, axis, intensity

    @staticmethod
    def get_axis_and_size(axis, array):
        if axis == 1:
            axis = 0
            size = array.shape[0]
        elif axis == 0:
            axis = 1
            size = array.shape[1]
        elif axis == 2:  # we will also traverse in sagittal (if RAS)
            size = array.shape[0]
        else:
            raise RuntimeError(f'Axis "{axis}" is not valid')
        return axis, size

    @staticmethod
    def get_slice(axis, array, slice_idx):
        # Comments apply if RAS
        if axis == 0:  # sagittal (columns) - artifact AP
            image_slice = array[slice_idx, ...]
        elif axis == 1:  # coronal (columns) - artifact LR
            image_slice = array[:, slice_idx, :]
        elif axis == 2:  # sagittal (rows) - artifact IS
            image_slice = array[slice_idx, ...].T
        else:
            raise RuntimeError(f'Axis "{axis}" is not valid')
        return image_slice

    def add_artifact(
            self,
            image: sitk.Image,
            num_ghosts: int,
            axis: int,
            intensity: float,
            ):
        array = sitk.GetArrayFromImage(image).transpose()
        # Leave first 5% of frequencies untouched. If the image is in RAS
        # orientation, this helps applying the ghosting in the desired axis
        # intuitively
        # [Why? I forgot]
        percentage_to_avoid = 0.05
        axis, size = self.get_axis_and_size(axis, array)
        for slice_idx in range(size):
            image_slice = self.get_slice(axis, array, slice_idx)
            spectrum = self.fourier_transform(image_slice)
            for row_idx, row in enumerate(spectrum):
                if row_idx % num_ghosts:
                    continue
                progress = row_idx / array.shape[0]
                if np.abs(progress - 0.5) < percentage_to_avoid / 2:
                    continue
                row *= 1 - intensity
            image_slice *= 0
            image_slice += self.inv_fourier_transform(spectrum)
        return array
