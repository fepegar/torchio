"""
Adapted from NiftyNet
"""

from pathlib import Path
from typing import Dict, Callable, Tuple, Sequence, Union, Optional
import torch
import numpy as np
import numpy.ma as ma
import nibabel as nib
from tqdm import tqdm
from ....torchio import DATA, TypePath, TypeCallable
from . import NormalizationTransform

DEFAULT_CUTOFF = 0.01, 0.99
STANDARD_RANGE = 0, 100


class HistogramStandardization(NormalizationTransform):
    """Perform histogram standardization of intensity values.

    See example in :py:func:`torchio.transforms.HistogramStandardization.train`.

    Args:
        landmarks_dict: Dictionary in which keys are image names in the sample
            and values are NumPy arrays defining the landmarks after training
            with :py:meth:`torchio.transforms.HistogramStandardization.train`.
        masking_method: See
            :py:class:`~torchio.transforms.preprocessing.normalization_transform.NormalizationTransform`.

    """
    def __init__(
            self,
            landmarks_dict: Dict[str, np.ndarray],
            masking_method: Union[str, TypeCallable, None] = None,
            ):
        super().__init__(masking_method=masking_method)
        self.landmarks_dict = landmarks_dict

    def apply_normalization(
            self,
            sample: dict,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:
        if image_name not in self.landmarks_dict:
            keys = tuple(self.landmarks_dict.keys())
            message = (
                f'Image name "{image_name}" should be a key in the'
                f' landmarks dictionary, whose keys are {keys}'
            )
            raise KeyError(message)
        image_dict = sample[image_name]
        landmarks = self.landmarks_dict[image_name]
        image_dict[DATA] = normalize(
            image_dict[DATA],
            landmarks,
            mask=mask,
        )

    @classmethod
    def train(
            cls,
            images_paths: Sequence[TypePath],
            cutoff: Optional[Tuple[float, float]] = None,
            mask_path: Optional[TypePath] = None,
            masking_function: Optional[Callable] = None,
            output_path: Optional[TypePath] = None,
            ) -> np.ndarray:
        """Extract average histogram landmarks from images used for training.

        Args:
            images_paths: List of image paths used to train.
            cutoff: Optional "minimum and maximum percentile values,
                respectively, that are used to select a range of intensity of
                interest".
                Equivalent to :math:`pc_1` and :math:`pc_2` in
                `Nyúl and Udupa's paper <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.204.102&rep=rep1&type=pdf>`_.
            mask_path: Optional path to a mask image to extract voxels used for
                training.
            masking_function: Optional function used to extract voxels used for
                training.
            output_path: Optional file path with extension ``.txt`` or
                ``.npy``, where the landmarks will be saved.

        Example:

            >>> from pathlib import Path
            >>> import numpy as np
            >>> from torchio.transforms import HistogramStandardization
            >>>
            >>> t1_paths = ['subject_a_t1.nii', 'subject_b_t1.nii.gz']
            >>> t2_paths = ['subject_a_t2.nii', 'subject_b_t2.nii.gz']
            >>>
            >>> t1_landmarks_path = Path('t1_landmarks.npy')
            >>> t2_landmarks_path = Path('t2_landmarks.npy')
            >>>
            >>> t1_landmarks = (
            ...     np.load(t1_landmarks_path)
            ...     if t1_landmarks_path.is_file()
            ...     else HistogramStandardization.train(t1_paths)
            ... )
            >>> t2_landmarks = (
            ...     np.load(t2_landmarks_path)
            ...     if t2_landmarks_path.is_file()
            ...     else HistogramStandardization.train(t2_paths)
            ... )
            >>>
            >>> landmarks_dict = {
            ...     't1': t1_landmarks,
            ...     't2': t2_landmarks,
            ... }
            >>>
            >>> transform = HistogramStandardization(landmarks_dict)
        """
        cutoff = DEFAULT_CUTOFF if cutoff is None else cutoff
        percentiles_database = []
        for index, image_file_path in enumerate(tqdm(images_paths)):
            data = nib.load(str(image_file_path)).get_fdata(dtype=np.float32)
            if masking_function is not None:
                mask = masking_function(data)
            else:
                if mask_path is not None:
                    mask = nib.load(str(mask_path)).get_fdata()
                    mask = mask > 0
                else:
                    mask = np.ones_like(data, dtype=np.bool)
            percentiles = _compute_percentiles(data, mask, cutoff)
            percentiles_database.append(percentiles)
        percentiles_database = np.vstack(percentiles_database)
        s1, s2 = STANDARD_RANGE
        mapping = _averaged_mapping(percentiles_database, s1, s2)

        if output_path is not None:
            output_path = Path(output_path).expanduser()
            extension = output_path.suffix
            if extension == '.txt':
                modality = 'image'
                text = f'{modality} {" ".join(map(str, mapping))}'
                output_path.write_text(text)
            elif extension == '.npy':
                np.save(output_path, mapping)

        return mapping



def _compute_percentiles(
        img: np.ndarray,
        mask: np.ndarray,
        cutoff: Tuple[float, float],
        ) -> np.ndarray:
    """
    Creates the list of percentile values to be used as landmarks for the
    linear fitting.

    :param img: Image on which to determine the percentiles
    :param mask: Mask to use over the image to constraint to the relevant
    information
    :param cutoff: Values of the minimum and maximum percentiles to use for
    the linear fitting
    :return perc_results: list of percentiles value for the given image over
    the mask
    """
    perc = [cutoff[0],
            0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9,
            cutoff[1]]
    masked_img = ma.masked_array(img, np.logical_not(mask)).compressed()
    perc_results = np.percentile(masked_img, 100 * np.array(perc))
    return perc_results


def _standardize_cutoff(cutoff: np.ndarray) -> np.ndarray:
    """Standardize the cutoff values given in the configuration.

    Computes percentile landmark normalization by default.

    """
    cutoff = np.asarray(cutoff)
    cutoff[0] = max(0., cutoff[0])
    cutoff[1] = min(1., cutoff[1])
    cutoff[0] = np.min([cutoff[0], 0.09])
    cutoff[1] = np.max([cutoff[1], 0.91])
    return cutoff


def _averaged_mapping(
        perc_database: np.ndarray,
        s1: float,
        s2: float,
        ) -> np.ndarray:
    """
    Map the landmarks of the database to the chosen range
    :param perc_database: perc_database over which to perform the averaging
    :param s1, s2: limits of the mapping range
    :return final_map: the average mapping
    """
    # assuming shape: n_data_points = perc_database.shape[0]
    #                 n_percentiles = perc_database.shape[1]
    slope = (s2 - s1) / (perc_database[:, -1] - perc_database[:, 0])
    slope = np.nan_to_num(slope)
    final_map = slope.dot(perc_database) / perc_database.shape[0]
    intercept = np.mean(s1 - slope * perc_database[:, 0])
    final_map = final_map + intercept
    return final_map


def normalize(
        tensor: torch.Tensor,
        landmarks: np.ndarray,
        mask: Optional[np.ndarray],
        cutoff: Optional[Tuple[float, float]] = None,
        epsilon: float = 1e-5,
        ) -> torch.Tensor:
    cutoff_ = DEFAULT_CUTOFF if cutoff is None else cutoff
    array = tensor.numpy()
    mapping = landmarks

    img = array
    image_shape = img.shape
    img = img.reshape(-1).astype(np.float32)

    if mask is None:
        mask = np.ones_like(img, np.bool)
    mask = mask.reshape(-1)

    range_to_use = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]

    cutoff_ = _standardize_cutoff(cutoff_)
    perc = _compute_percentiles(img, mask, cutoff_)

    # Apply linear histogram standardization
    range_mapping = mapping[range_to_use]
    range_perc = perc[range_to_use]
    diff_mapping = range_mapping[1:] - range_mapping[:-1]
    diff_perc = range_perc[1:] - range_perc[:-1]

    # handling the case where two landmarks are the same
    # for a given input image. This usually happens when
    # image background is not removed from the image.
    diff_perc[diff_perc < epsilon] = np.inf

    affine_map = np.zeros([2, len(range_to_use) - 1])
    # compute slopes of the linear models
    affine_map[0] = diff_mapping / diff_perc
    # compute intercepts of the linear models
    affine_map[1] = range_mapping[:-1] - affine_map[0] * range_perc[:-1]

    bin_id = np.digitize(img, range_perc[1:-1], right=False)
    lin_img = affine_map[0, bin_id]
    aff_img = affine_map[1, bin_id]
    new_img = lin_img * img + aff_img
    new_img = new_img.reshape(image_shape)
    new_img = new_img.astype(np.float32)
    new_img = torch.from_numpy(new_img)
    return new_img


train = train_histogram = HistogramStandardization.train
