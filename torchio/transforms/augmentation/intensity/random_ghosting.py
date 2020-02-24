import warnings
from typing import Tuple, Optional, Union
import torch
import numpy as np
import SimpleITK as sitk
from ....utils import is_image_dict
from ....torchio import INTENSITY, DATA, AFFINE
from .. import RandomTransform


class RandomGhosting(RandomTransform):
    def __init__(
            self,
            num_ghosts_range: Union[int, Tuple[int, int]] = (4, 10),
            axes: Tuple[int, ...] = (0, 1, 2),
            proportion_to_augment: float = 1,
            seed: Optional[int] = None,
            verbose: bool = False,
            ):
        super().__init__(seed=seed, verbose=verbose)
        self.proportion_to_augment = self.parse_probability(
            proportion_to_augment,
            'proportion_to_augment',
        )
        self.axes = axes
        if isinstance(num_ghosts_range, int):
            self.num_ghosts_range = num_ghosts_range, num_ghosts_range
        else:
            self.num_ghosts_range = num_ghosts_range

    def apply_transform(self, sample: dict) -> dict:
        for image_name, image_dict in sample.items():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] != INTENSITY:
                continue
            params = self.get_params(
                self.num_ghosts_range,
                self.axes,
                self.proportion_to_augment,
            )
            num_ghosts_param, axis_param, do_it = params
            sample[image_name]['random_ghosting_axis'] = axis_param
            sample[image_name]['random_ghosting_num_ghosts'] = num_ghosts_param
            sample[image_name]['random_ghosting_do'] = do_it
            if not do_it:
                return sample
            if (image_dict[DATA][0] < -0.1).any():
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
                image_dict[DATA][0],
                image_dict[AFFINE],
            )
            image_dict[DATA] = self.add_artifact(
                image,
                num_ghosts_param,
                axis_param,
            )
            # Add channels dimension
            image_dict[DATA] = image_dict[DATA][np.newaxis, ...]
            image_dict[DATA] = torch.from_numpy(image_dict[DATA])
        return sample

    @staticmethod
    def get_params(
            num_ghosts_range: Tuple[int, int],
            axes: Tuple[int, ...],
            probability: float,
            ) -> Tuple:
        ng_min, ng_max = num_ghosts_range
        num_ghosts_param = torch.randint(ng_min, ng_max + 1, (1,)).item()
        axis_param = axes[torch.randint(0, len(axes), (1,))]
        do_it = torch.rand(1) < probability
        return num_ghosts_param, axis_param, do_it

    def add_artifact(
            self,
            image: sitk.Image,
            num_ghosts: int,
            axis: int,
            ):
        array = sitk.GetArrayFromImage(image).transpose()
        percentage_to_avoid = 0.05  # Leave first 5% of frequencies untouched
        # If the image is in RAS orientation, this helps applying the ghosting
        # in the desired axis intuitively
        if axis == 1:
            axis = 0
            size = array.shape[0]
        elif axis == 0:
            axis = 1
            size = array.shape[1]
        elif axis == 2:  # we will also traverse in sagittal (if RAS)
            size = array.shape[0]

        for slice_idx in range(size):
            # Comments apply if RAS
            if axis == 0:  # sagittal (columns) - artifact AP
                slice_ = array[slice_idx, ...]
            elif axis == 1:  # coronal (columns) - artifact LR
                slice_ = array[:, slice_idx, :]
            elif axis == 2:  # sagittal (rows) - artifact IS
                slice_ = array[slice_idx, ...].T
            spectrum = self.fourier_transform(slice_)
            for row_idx, row in enumerate(spectrum):
                if row_idx % num_ghosts:
                    continue
                progress = row_idx / array.shape[0]
                if np.abs(progress - 0.5) < percentage_to_avoid / 2:
                    continue
                row *= 0
            slice_ *= 0
            slice_ += self.inv_fourier_transform(spectrum)

        return array
