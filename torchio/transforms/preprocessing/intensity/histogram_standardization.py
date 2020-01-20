"""
Adapted from NiftyNet
"""

from pathlib import Path
import torch
import numpy as np
import numpy.ma as ma
import nibabel as nib
from tqdm import tqdm
from ....torchio import DATA
from .normalization_transform import NormalizationTransform

DEFAULT_CUTOFF = 0.01, 0.99
STANDARD_RANGE = 0, 100


class HistogramStandardization(NormalizationTransform):
    def __init__(self, landmarks_dict, masking_method=None, verbose=False):
        super().__init__(masking_method=masking_method, verbose=verbose)
        self.landmarks_dict = landmarks_dict

    def apply_normalization(self, sample, image_name, mask):
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


def __compute_percentiles(img, mask, cutoff):
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


def __standardize_cutoff(cutoff, type_hist='percentile'):
    """
    Standardizes the cutoff values given in the configuration

    :param cutoff:
    :param type_hist: Type of landmark normalisation chosen (median,
    quartile, percentile)
    :return cutoff: cutoff with appropriate adapted values
    """
    cutoff = np.asarray(cutoff)
    if cutoff is None:
        return DEFAULT_CUTOFF
    if len(cutoff) > 2:
        cutoff = np.unique([np.min(cutoff), np.max(cutoff)])
    if len(cutoff) < 2:
        return DEFAULT_CUTOFF
    if cutoff[0] > cutoff[1]:
        cutoff[0], cutoff[1] = cutoff[1], cutoff[0]
    cutoff[0] = max(0., cutoff[0])
    cutoff[1] = min(1., cutoff[1])
    if type_hist == 'quartile':
        cutoff[0] = np.min([cutoff[0], 0.24])
        cutoff[1] = np.max([cutoff[1], 0.76])
    else:
        cutoff[0] = np.min([cutoff[0], 0.09])
        cutoff[1] = np.max([cutoff[1], 0.91])
    return cutoff


def __averaged_mapping(perc_database, s1, s2):
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
        data,
        landmarks,
        mask=None,
        cutoff=DEFAULT_CUTOFF,
        epsilon=1e-5,
        ):
    data = data.numpy()
    mapping = landmarks

    img = data
    image_shape = img.shape
    img = img.reshape(-1).astype(np.float32)

    if mask is None:
        mask = np.ones_like(img, np.bool)
    mask = mask.reshape(-1)

    range_to_use = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]

    cutoff = __standardize_cutoff(cutoff)
    perc = __compute_percentiles(img, mask, cutoff)

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


def train(
        images_paths,
        cutoff=None,
        mask_path=None,
        masking_function=None,
        output_path=None,
        ):
    """
    Output path extension should be .txt or .npy
    """
    cutoff = DEFAULT_CUTOFF if cutoff is None else cutoff
    percentiles_database = []
    for index, image_file_path in enumerate(tqdm(images_paths)):
        # NiftyNet implementation says image should be float
        data = nib.load(str(image_file_path)).get_fdata(dtype=np.float32)

        if masking_function is not None:
            mask = masking_function(data)
        else:
            if mask_path is not None:
                mask = nib.load(mask_path[index]).get_fdata()
                mask = mask > 0
            else:
                mask = np.ones_like(data, dtype=np.bool)
        percentiles = __compute_percentiles(data, mask, cutoff)
        percentiles_database.append(percentiles)
    percentiles_database = np.vstack(percentiles_database)
    s1, s2 = STANDARD_RANGE
    mapping = __averaged_mapping(percentiles_database, s1, s2)

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
