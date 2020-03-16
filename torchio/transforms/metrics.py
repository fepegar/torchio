import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import time


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        if img1.size()[0] == 1:
            res = ssim_map.mean()
        else:
            res = ssim_map.mean(axis=list(range(1, img1.ndim))) #one value per patch
        return res
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D_dist(img1, img2, window, window_size, channel, aggregate="avg"):

    if len(img1.size()) == 4: #missing batch dim
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    s1 = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    s2 = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    d1 = torch.sqrt(1 - s1)
    d2 = torch.sqrt(1 - s2)
    d1[torch.isnan(d1)] = 0
    d2[torch.isnan(d2)] = 0

    if aggregate.lower() == "normed":
        res = torch.norm(torch.sqrt(d1 ** 2 + d2 ** 2), 2)
    else:
        res = torch.mean(torch.sqrt(d1 ** 2 + d2 ** 2))

    return res


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=3, size_average=True, distance=1):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)
        self.distance = distance

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        if self.distance==1:
            res =  1 - _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)
        else :
            res = _ssim_3D_dist(img1, img2, window, self.window_size, channel)

        return res

def ssim3D(img1, img2, window_size=3, size_average=True, verbose=False):

    if verbose:
        start = time.time()

    if len(img1.size()) == 4: #missing batch dim
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    #img1 = img1.float()
    #img2 = img2.float()

    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    res =  _ssim_3D(img1, img2, window, window_size, channel, size_average)

    if verbose:
        duration = time.time() - start
        print(f'Ssim calculation : {duration:.3f} seconds')

    return res
########################################################################################################################


def th_pearsonr(x, y):
    """
    mimics scipy.stats.pearsonr
    """
    x = torch.flatten(x)
    y = torch.flatten(y)

    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val
########################################################################################################################


def nrmse(image_true, image_test, normalization="euclidean"):
    '''
    A Pytorch version of scikit-image's implementation of normalized_root_mse
    https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.normalized_root_mse
    Compute the normalized root mean-squared error (NRMSE) between two
    images.
    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    normalization : {'euclidean', 'min-max', 'mean'}, optional
        Controls the normalization method to use in the denominator of the
        NRMSE.  There is no standard method of normalization across the
        literature [1]_.  The methods available here are as follows:
        - 'euclidean' : normalize by the averaged Euclidean norm of
          ``im_true``::
              NRMSE = RMSE * sqrt(N) / || im_true ||
          where || . || denotes the Frobenius norm and ``N = im_true.size``.
          This result is equivalent to::
              NRMSE = || im_true - im_test || / || im_true ||.
        - 'min-max'   : normalize by the intensity range of ``im_true``.
        - 'mean'      : normalize by the mean of ``im_true``

    Returns
    -------
    nrmse : float
        The NRMSE metric.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Root-mean-square_deviation

    '''

    normalization = normalization.lower()

    if normalization == "min-max":
        denom = image_true.max() - image_true.min()
    elif normalization == "mean":
        denom = image_true.mean()
    else:
        if normalization != "euclidean":
            raise Warning("Unsupported norm type. Found {}.\nUsing euclidean by default".format(normalization))
        denom = torch.sqrt(torch.mean(image_true ** 2))

    return (F.mse_loss(image_true, image_test).sqrt())/denom



#code from https://github.com/rogerberm/pytorch-ncc/blob/master/NCC.py

def patch_mean(images, patch_shape):
    """
    Computes the local mean of an image or set of images.
    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)
    Returns:
        Tensor same size as the image, with local means computed independently for each channel.
    Example::
        >>> images = torch.randn(4, 3, 15, 15)           # 4 images, 3 channels, 15x15 pixels each
        >>> patch_shape = 3, 5, 5                        # 3 channels, 5x5 pixels neighborhood
        >>> means = patch_mean(images, patch_shape)
        >>> expected_mean = images[3, 2, :5, :5].mean()  # mean of the third image, channel 2, top left 5x5 patch
        >>> computed_mean = means[3, 2, 5//2, 5//2]      # computed mean whose 5x5 neighborhood covers same patch
        >>> computed_mean.isclose(expected_mean).item()
        1
    """
    channels, *patch_size = patch_shape
    dimensions = len(patch_size)
    padding = tuple(side // 2 for side in patch_size)

    conv = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]

    # Convolution with these weights will effectively compute the channel-wise means
    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full((channels, channels, *patch_size), fill_value=1 / patch_elements)
    weights = weights.to(images.device)

    # Make convolution operate on single channels
    channel_selector = torch.eye(channels).byte()
    weights[1 - channel_selector] = 0

    result = conv(images, weights, padding=padding, bias=None)

    return result


def patch_std(image, patch_shape):
    """
    Computes the local standard deviations of an image or set of images.
    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)
    Returns:
        Tensor same size as the image, with local standard deviations computed independently for each channel.
    Example::
        >>> images = torch.randn(4, 3, 15, 15)           # 4 images, 3 channels, 15x15 pixels each
        >>> patch_shape = 3, 5, 5                        # 3 channels, 5x5 pixels neighborhood
        >>> stds = patch_std(images, patch_shape)
        >>> patch = images[3, 2, :5, :5]
        >>> expected_std = patch.std(unbiased=False)     # standard deviation of the third image, channel 2, top left 5x5 patch
        >>> computed_std = stds[3, 2, 5//2, 5//2]        # computed standard deviation whose 5x5 neighborhood covers same patch
        >>> computed_std.isclose(expected_std).item()
        1
    """
    return (patch_mean(image**2, patch_shape) - patch_mean(image, patch_shape)**2).sqrt()


def channel_normalize(template):
    """
    Z-normalize image channels independently.
    """
    reshaped_template = template.clone().view(template.shape[0], -1)
    reshaped_template.sub_(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div_(reshaped_template.std(dim=-1, keepdim=True, unbiased=False))

    return reshaped_template.view_as(template)


class NCC(torch.nn.Module):
    """
    Computes the [Zero-Normalized Cross-Correlation][1] between an image and a template.
    Example:
        >>> lena_path = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
        >>> lena_tensor = torch.Tensor(plt.imread(lena_path)).permute(2, 0, 1).cuda()
        >>> patch_center = 275, 275
        >>> y1, y2 = patch_center[0] - 25, patch_center[0] + 25
        >>> x1, x2 = patch_center[1] - 25, patch_center[1] + 25
        >>> lena_patch = lena_tensor[:, y1:y2 + 1, x1:x2 + 1]
        >>> ncc = NCC(lena_patch)
        >>> ncc_response = ncc(lena_tensor[None, ...])
        >>> ncc_response.max()
        tensor(1.0000, device='cuda:0')
        >>> np.unravel_index(ncc_response.argmax(), lena_tensor.shape)
        (0, 275, 275)
    [1]: https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
    """
    def __init__(self, template, keep_channels=False):
        super().__init__()

        self.keep_channels = keep_channels

        channels, *template_shape = template.shape
        dimensions = len(template_shape)
        self.padding = tuple(side // 2 for side in template_shape)

        self.conv_f = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]
        self.normalized_template = channel_normalize(template)
        ones = template.dim() * (1, )
        self.normalized_template = self.normalized_template.repeat(channels, *ones)
        # Make convolution operate on single channels
        channel_selector = torch.eye(channels).byte()
        self.normalized_template[1 - channel_selector] = 0
        # Reweight so that output is averaged
        patch_elements = torch.Tensor(template_shape).prod().item()
        self.normalized_template.div_(patch_elements)

    def forward(self, image):
        result = self.conv_f(image, self.normalized_template, padding=self.padding, bias=None)
        std = patch_std(image, self.normalized_template.shape[1:])
        result.div_(std)
        if not self.keep_channels:
            result = result.mean(dim=1)

        return result





#code from https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
'''
Non-parametric computation of entropy and mutual-information
Adapted by G Varoquaux for code created by R Brette, itself
from several papers (see in the code).
These computations rely on nearest-neighbor statistics
'''
import numpy as np

from scipy.special import gamma,psi
from scipy import ndimage
from scipy.linalg import det
from numpy import pi

from sklearn.neighbors import NearestNeighbors

__all__=['entropy', 'mutual_information', 'entropy_gaussian']

EPS = np.finfo(float).eps


def nearest_distances(X, k=1):
    '''
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    '''
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor


def entropy_gaussian(C):
    '''
    Entropy of a gaussian variable with covariance matrix C
    '''
    if np.isscalar(C): # C is the variance
        return .5*(1 + np.log(2*pi)) + .5*np.log(C)
    else:
        n = C.shape[0] # dimension
        return .5*n*(1 + np.log(2*pi)) + .5*np.log(abs(det(C)))


def entropy(X, k=1):
    ''' Returns the entropy of the X.
    Parameters
    ===========
    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation
    Notes
    ======
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''

    # Distance to kth nearest neighbor
    r = nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1)
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


def mutual_information(variables, k=1):
    '''
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n_samples, n_features)
    where
      n = number of samples
      dx,dy = number of dimensions
    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation
    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
    '''
    if len(variables) < 2:
        raise AttributeError(
                "Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    # check that mi(X, X) = entropy(X)
    check = np.unique(all_vars, axis=1)
    if all_vars.shape[1] != check.shape[1]:
        print(f'WARNING: dropping {all_vars.shape[1] - check.shape[1]} variables as the samples are identical!')
        all_vars = check
    return (sum([entropy(X, k=k) for X in variables])
            - entropy(all_vars, k=k))


def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                 output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi



###############################################################################
# Tests

def test_entropy():
    # Testing against correlated Gaussian variables
    # (analytical results are known)
    # Entropy of a 3-dimensional gaussian variable
    rng = np.random.RandomState(0)
    n = 50000
    d = 3
    P = np.array([[1, 0, 0], [0, 1, .5], [0, 0, 1]])
    C = np.dot(P, P.T)
    Y = rng.randn(d, n)
    X = np.dot(P, Y)
    H_th = entropy_gaussian(C)
    H_est = entropy(X.T, k=5)
    # Our estimated entropy should always be less that the actual one
    # (entropy estimation undershoots) but not too much
    np.testing.assert_array_less(H_est, H_th)
    np.testing.assert_array_less(.9*H_th, H_est)


def test_mutual_information():
    # Mutual information between two correlated gaussian variables
    # Entropy of a 2-dimensional gaussian variable
    n = 50000
    rng = np.random.RandomState(0)
    #P = np.random.randn(2, 2)
    P = np.array([[1, 0], [0.5, 1]])
    C = np.dot(P, P.T)
    U = rng.randn(2, n)
    Z = np.dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    # in bits
    MI_est = mutual_information((X, Y), k=5)
    MI_th = (entropy_gaussian(C[0, 0])
             + entropy_gaussian(C[1, 1])
             - entropy_gaussian(C)
            )
    # Our estimator should undershoot once again: it will undershoot more
    # for the 2D estimation that for the 1D estimation
    print((MI_est, MI_th))
    np.testing.assert_array_less(MI_est, MI_th)
    np.testing.assert_array_less(MI_th, MI_est  + .3)


def test_degenerate():
    # Test that our estimators are well-behaved with regards to
    # degenerate solutions
    rng = np.random.RandomState(0)
    x = rng.randn(50000)
    X = np.c_[x, x]
    assert np.isfinite(entropy(X))
    assert np.isfinite(mutual_information((x[:, np.newaxis],
                                           x[:,  np.newaxis])))
    assert 2.9 < mutual_information_2d(x, x) < 3.1


def test_mutual_information_2d():
    # Mutual information between two correlated gaussian variables
    # Entropy of a 2-dimensional gaussian variable
    n = 50000
    rng = np.random.RandomState(0)
    #P = np.random.randn(2, 2)
    P = np.array([[1, 0], [.9, .1]])
    C = np.dot(P, P.T)
    U = rng.randn(2, n)
    Z = np.dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    # in bits
    MI_est = mutual_information_2d(X.ravel(), Y.ravel())
    MI_th = (entropy_gaussian(C[0, 0])
             + entropy_gaussian(C[1, 1])
             - entropy_gaussian(C)
            )
    print((MI_est, MI_th))
    # Our estimator should undershoot once again: it will undershoot more
    # for the 2D estimation that for the 1D estimation
    np.testing.assert_array_less(MI_est, MI_th)
    np.testing.assert_array_less(MI_th, MI_est  + .2)


if __name__ == '__main__':
    # Run our tests
    test_entropy()
    test_mutual_information()
    test_degenerate()
    test_mutual_information_2d()
