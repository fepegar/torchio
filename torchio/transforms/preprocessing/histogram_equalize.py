import torch
import numpy as np
from ...torchio import DATA
from .normalization_transform import NormalizationTransform
from warnings import warn

#taken from https://github.com/scikit-image/scikit-image/blob/master/skimage/exposure/exposure.py

class HistogramEqualize(NormalizationTransform):
    def __init__(
            self,
            out_min_max=(-1, 1),
            percentiles=(1, 99),
            masking_method=None,
            nbins=256,
            verbose=False,
            ):
        super().__init__(masking_method=masking_method, verbose=verbose)
        self.out_min, self.out_max = out_min_max
        self.percentiles = percentiles
        self.nbins = nbins

    def apply_normalization(self, sample, image_name, mask):
        """
        This could probably be written in two or three lines
        """
        image_dict = sample[image_name]
        image_dict[DATA] = self.rescale(image_dict[DATA], mask)

    def rescale(self, data, mask):
        array = data.numpy()
        mask = mask.numpy()
        values = array[mask]

        cdf, bin_centers = cumulative_distribution(values, self.nbins)
        out = np.interp(array.flat, bin_centers, cdf)
        return torch.from_numpy( out.reshape(array.shape) )


def cumulative_distribution(image, nbins=256):
    """Return cumulative distribution function (cdf) for the given image.
    Parameters
    ----------
    image : array
        Image array.
    nbins : int, optional
        Number of bins for image histogram.
    Returns
    -------
    img_cdf : array
        Values of cumulative distribution function.
    bin_centers : array
        Centers of bins.
    See Also
    --------
    histogram
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cumulative_distribution_function
    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> image = img_as_float(data.camera())
    >>> hi = exposure.histogram(image)
    >>> cdf = exposure.cumulative_distribution(image)
    >>> np.alltrue(cdf[0] == np.cumsum(hi[0])/float(image.size))
    True
    """
    hist, bin_centers = histogram(image, nbins)
    img_cdf = hist.cumsum()
    img_cdf = img_cdf / float(img_cdf[-1])
    return img_cdf, bin_centers

def histogram(image, nbins=256, source_range='image', normalize=False):
    """Return histogram of image.
    Unlike `numpy.histogram`, this function returns the centers of bins and
    does not rebin integer arrays. For integer arrays, each integer value has
    its own bin, which improves speed and intensity-resolution.
    The histogram is computed on the flattened image: for color images, the
    function should be used separately on each channel to obtain a histogram
    for each color channel.
    Parameters
    ----------
    image : array
        Input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    source_range : string, optional
        'image' (default) determines the range from the input image.
        'dtype' determines the range from the expected range of the images
        of that data type.
    normalize : bool, optional
        If True, normalize the histogram by the sum of its values.
    Returns
    -------
    hist : array
        The values of the histogram.
    bin_centers : array
        The values at the center of the bins.
    See Also
    --------
    cumulative_distribution
    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> image = img_as_float(data.camera())
    >>> np.histogram(image, bins=2)
    (array([107432, 154712]), array([0. , 0.5, 1. ]))
    >>> exposure.histogram(image, nbins=2)
    (array([107432, 154712]), array([0.25, 0.75]))
    """
    sh = image.shape
    if len(sh) == 3 and sh[-1] < 4:
        warn("This might be a color image. The histogram will be "
             "computed on the flattened image. You can instead "
             "apply this function to each color channel.")

    image = image.flatten()
    # For integer types, histogramming with bincount is more efficient.
    if np.issubdtype(image.dtype, np.integer):
        hist, bin_centers = _bincount_histogram(image, source_range)
    else:
        if source_range == 'image':
            hist_range = None
        #elif source_range == 'dtype':
        #    hist_range = dtype_limits(image, clip_negative=False)
        else:
            ValueError('Wrong value for the `source_range` argument')
        hist, bin_edges = np.histogram(image, bins=nbins, range=hist_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    if normalize:
        hist = hist / np.sum(hist)
    return hist, bin_centers

def _bincount_histogram(image, source_range):
    """
    Efficient histogram calculation for an image of integers.
    This function is significantly more efficient than np.histogram but
    works only on images of integers. It is based on np.bincount.
    Parameters
    ----------
    image : array
        Input image.
    source_range : string
        'image' determines the range from the input image.
        'dtype' determines the range from the expected range of the images
        of that data type.
    Returns
    -------
    hist : array
        The values of the histogram.
    bin_centers : array
        The values at the center of the bins.
    """
    if source_range not in ['image', 'dtype']:
        raise ValueError('Incorrect value for `source_range` argument: {}'.format(source_range))
    if source_range == 'image':
        image_min = int(image.min().astype(np.int64))
        image_max = int(image.max().astype(np.int64))
    #elif source_range == 'dtype':
    #    image_min, image_max = dtype_limits(image, clip_negative=False)
    image, offset = _offset_array(image, image_min, image_max)
    hist = np.bincount(image.ravel(), minlength=image_max - image_min + 1)
    bin_centers = np.arange(image_min, image_max + 1)
    if source_range == 'image':
        idx = max(image_min, 0)
        hist = hist[idx:]
    return hist, bin_centers

def _offset_array(arr, low_boundary, high_boundary):
    """Offset the array to get the lowest value at 0 if negative."""
    if low_boundary < 0:
        offset = low_boundary
        dyn_range = high_boundary - low_boundary
        # get smallest dtype that can hold both minimum and offset maximum
        offset_dtype = np.promote_types(np.min_scalar_type(dyn_range),
                                        np.min_scalar_type(low_boundary))
        if arr.dtype != offset_dtype:
            # prevent overflow errors when offsetting
            arr = arr.astype(offset_dtype)
        arr = arr - offset
    else:
        offset = 0
    return arr, offset
