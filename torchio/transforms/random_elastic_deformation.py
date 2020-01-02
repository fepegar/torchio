import warnings
import torch
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
        # Only do augmentation with a probability `proportion_to_augment`
        do_augmentation = torch.rand(1) < self.proportion_to_augment
        if not do_augmentation:
            return sample

        bspline_params = None
        nothing_resampled = True
        for image_dict in sample.values():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] == LABEL:
                interpolation = Interpolation.NEAREST
            else:
                interpolation = self.image_interpolation
            # TODO: assert that all images have the same shape
            if bspline_params is None:
                image = self.nib_to_sitk(
                    image_dict['data'].squeeze(), image_dict['affine'])
                bspline_params = self.get_params(
                    image,
                    self.num_control_points,
                    self.deformation_std,
                )
            image_dict['data'] = self.apply_bspline_transform(
                image_dict['data'],
                image_dict['affine'],
                bspline_params,
                interpolation,
            )
            nothing_resampled = False
        if nothing_resampled:
            warnings.warn(
                'No images were resampled.'
                f' Sample keys: {sample.keys()}'
            )
        sample['random_elastic_deformation'] = bspline_params
        return sample

    @staticmethod
    def get_params(image, num_control_points, deformation_std):
        mesh_shape = 3 * (num_control_points,)
        bspline_transform = sitk.BSplineTransformInitializer(image, mesh_shape)
        default_params = bspline_transform.GetParameters()
        bspline_params = torch.rand(len(default_params)) * deformation_std
        return bspline_params.numpy()

    @staticmethod
    def get_bspline_transform(image, num_control_points, bspline_params):
        mesh_shape = 3 * (num_control_points,)
        bspline_transform = sitk.BSplineTransformInitializer(image, mesh_shape)
        bspline_transform.SetParameters(bspline_params.tolist())
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
                image,
                self.num_control_points,
                bspline_params,
            )

            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(interpolation.value)
            resampler.SetReferenceImage(image)
            resampler.SetDefaultPixelValue(0)  # should I change this?
            resampler.SetTransform(bspline_transform)
            resampled = resampler.Execute(image)

            channel_array = sitk.GetArrayFromImage(resampled)
            channel_array = channel_array.transpose()  # ITK to NumPy
            array[i] = channel_array
        return array
