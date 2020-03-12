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
        return ssim_map.mean()
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
