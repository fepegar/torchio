import torch
import numpy as np
import SimpleITK as sitk
from ..torchio import LABEL
from ..utils import is_image_dict
from .interpolation import Interpolation
from .random_transform import RandomTransform


class RandomElasticDeformation(RandomTransform):
    def __init__(
            self,
            num_control_points=4,
            deformation_std=15,
            proportion_to_augment=0.5,
            spatial_rank=3,
            image_interpolation=Interpolation.LINEAR,
            seed=None,
            verbose=False,
            ):
        super().__init__(seed=seed, verbose=verbose)
        self._bspline_transformation = None
        self.num_control_points = max(num_control_points, 2)
        self.deformation_std = max(deformation_std, 1)
        self.proportion_to_augment = proportion_to_augment
        self.spatial_rank = spatial_rank
        self.image_interpolation = image_interpolation
        self.seed = seed
        self.verbose = verbose

    def apply_transform(self, sample):
        bspline_params = self.get_params()

        # only do augmentation with a probability `proportion_to_augment`
        do_augmentation = torch.rand(1) < self.proportion_to_augment
        if not do_augmentation:
            return sample

        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] == LABEL:
                interpolation = Interpolation.NEAREST
            else:
                interpolation = self.image_interpolation
            image_dict['data'] = self.apply_bspline_transform(
                image_dict['data'],
                image_dict['affine'],
                bspline_params,
                interpolation,
            )
        return sample

    @staticmethod
    def get_params():
        pass

    @staticmethod
    def get_bspline_transform(shape, deformation_std, num_control_points):
        shape = list(reversed(shape))
        shape[0], shape[2] = shape[2], shape[0]
        itkimg = sitk.GetImageFromArray(np.zeros(shape))
        trans_from_domain_mesh_size = 3 * [num_control_points]
        bspline_transform = sitk.BSplineTransformInitializer(
            itkimg, trans_from_domain_mesh_size)
        params = bspline_transform.GetParameters()
        params_numpy = np.asarray(params, dtype=float)
        params_numpy = params_numpy + np.random.randn(
            params_numpy.shape[0]) * deformation_std
        params = tuple(params_numpy)
        bspline_transform.SetParameters(params)
        return bspline_transform

    def apply_bspline_transform(
            self,
            array,
            affine,
            bspline_params,
            interpolation: Interpolation,
            ):
        if array.ndim != 4:
            message = (
                'Only 4D images (channels, i, j, k) are supported,'
                f' not {array.shape}'
            )
            raise NotImplementedError(message)
        for i, channel_array in enumerate(array):  # use sitk.VectorImage?
            image = self.nib_to_sitk(channel_array, affine)
            bspline_transform = self.get_bspline_transform(
                channel_array.shape,
                self.deformation_std,
                self.num_control_points,
            )

            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(interpolation.value)
            resampler.SetReferenceImage(image)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(bspline_transform)
            resampled = resampler.Execute(image)

            channel_array = sitk.GetArrayFromImage(resampled)
            channel_array = channel_array.transpose(2, 1, 0)  # ITK to NumPy
            array[i] = channel_array
        return array
