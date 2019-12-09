import enum
import torch
import numpy as np
import SimpleITK as sitk


class RandomElasticDeformation:
    """
    generate randomised elastic deformations
    along each dim for data augmentation
    """

    def __init__(self, num_controlpoints=4, std_deformation_sigma=15, proportion_to_augment=0.5, spatial_rank=3,
                 seed=None, verbose=False):
        """
        This layer elastically deforms the inputs,
        for data-augmentation purposes.

        :param num_controlpoints:
        :param std_deformation_sigma:
        :param proportion_to_augment: what fraction of the images
            to do augmentation on
        :param name: name for tensorflow graph
        (may be computationally expensive).
        """

        self._bspline_transformation = None
        self.num_controlpoints = max(num_controlpoints, 2)
        self.std_deformation_sigma = max(std_deformation_sigma, 1)
        self.proportion_to_augment = proportion_to_augment
        if not sitk:
            self.proportion_to_augment = -1
        self.spatial_rank = spatial_rank
        self.seed = seed
        self.verbose = verbose

    def __call__(self, sample, interp_orders=0):
        self.check_seed()

        self._randomise_bspline_transformation(sample['image'].shape)

        # only do augmentation with a probability `proportion_to_augment`
        # do_augmentation = np.random.rand() < self.proportion_to_augment
        # if not do_augmentation:
        #     return inputs

        if self.verbose:
            import time
            start = time.time()

        for key in 'image', 'label', 'sampler':
            if key not in sample:
                continue

            array = sample[key]
            for i, channel_array in enumerate(array):
                array[i] = self._apply_bspline_transformation(channel_array )
            sample[key] = array


        if self.verbose:
            duration = time.time() - start
            print(f'RandomAffine: {duration:.1f} seconds')
        return sample


    def check_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)

    def _randomise_bspline_transformation(self, shape):
        # generate transformation

        squeezed_shape = shape[1:]

        itkimg = sitk.GetImageFromArray(np.zeros(squeezed_shape))
        trans_from_domain_mesh_size = \
            [self.num_controlpoints] * itkimg.GetDimension()
        self._bspline_transformation = sitk.BSplineTransformInitializer(
            itkimg, trans_from_domain_mesh_size)

        params = self._bspline_transformation.GetParameters()
        params_numpy = np.asarray(params, dtype=float)
        params_numpy = params_numpy + np.random.randn(
            params_numpy.shape[0]) * self.std_deformation_sigma

        # remove z deformations! The resolution in z is too bad
        # params_numpy[0:int(len(params) / 3)] = 0

        params = tuple(params_numpy)
        self._bspline_transformation.SetParameters(params)


    def _apply_bspline_transformation(self, image, interp_order=3):
        """
        Apply randomised transformation to 2D or 3D image

        :param image: 2D or 3D array
        :param interp_order: order of interpolation
        :return: the transformed image
        """
        resampler = sitk.ResampleImageFilter()
        if interp_order > 1:
            resampler.SetInterpolator(sitk.sitkBSpline)
        elif interp_order == 1:
            resampler.SetInterpolator(sitk.sitkLinear)
        elif interp_order == 0:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            return image

        squeezed_image = np.squeeze(image)
        while squeezed_image.ndim < self.spatial_rank:
            # pad to the required number of dimensions
            squeezed_image = squeezed_image[..., None]
        sitk_image = sitk.GetImageFromArray(squeezed_image)

        resampler.SetReferenceImage(sitk_image)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(self._bspline_transformation)
        out_img_sitk = resampler.Execute(sitk_image)
        out_img = sitk.GetArrayFromImage(out_img_sitk)
        return out_img.reshape(image.shape)

