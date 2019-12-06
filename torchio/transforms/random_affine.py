import enum
import torch
import numpy as np
import SimpleITK as sitk


class Interpolation(enum.Enum):
    """
    TODO: add more
    """
    NEAREST = 0
    LINEAR = 1


class RandomAffine:
    def __init__(
            self,
            scales,
            angles,
            isotropic=False,
            seed=None,
            verbose=False,
            ):
        """
        Example:
        scales = (0.9, 1.2)
        angles = (-12, 12)  degrees
        """
        self.scales = scales
        self.angles = angles
        self.isotropic = isotropic
        self.seed = seed
        self.verbose = verbose

    def __call__(self, sample):
        self.check_seed()
        if self.verbose:
            import time
            start = time.time()
        scaling_params, rotation_params = self.get_params(
            self.scales, self.angles, self.isotropic)
        sample['random_scaling'] = scaling_params
        sample['random_rotation'] = rotation_params
        for key in 'image', 'label', 'sampler':
            if key == 'image':
                interpolation = Interpolation.LINEAR
            else:
                interpolation = Interpolation.NEAREST
            if key not in sample:
                continue
            array = sample[key]
            array = self.apply_transform(
                array,
                sample['affine'],
                scaling_params,
                rotation_params,
                interpolation,
            )
            sample[key] = array
        if self.verbose:
            duration = time.time() - start
            print(f'RandomAffine: {duration:.1f} seconds')
        return sample

    @staticmethod
    def get_params(scales, angles, isotropic):
        scaling_params = torch.FloatTensor(3).uniform_(*scales).tolist()
        if isotropic:
            scaling_params = 3 * scaling_params[0]
        rotation_params = torch.FloatTensor(3).uniform_(*angles).tolist()
        return scaling_params, rotation_params

    def check_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)

    @staticmethod
    def get_scaling_transform(scaling_params):
        """
        scaling_params are inverted so that they are more intuitive
        For example, 1.5 means the objects look 1.5 times larger
        """
        transform = sitk.ScaleTransform(3)
        scaling_params = 1 / np.array(scaling_params)
        transform.SetScale(scaling_params)
        return transform

    @staticmethod
    def get_rotation_transform(rotation_params):
        """
        rotation_params is in degrees
        """
        transform = sitk.Euler3DTransform()
        rotation_params = np.radians(rotation_params)
        transform.SetRotation(*rotation_params)
        return transform

    def apply_transform(
            self,
            array,  # assume 4D
            affine,
            scaling_params,
            rotation_params,
            interpolation: Interpolation,
            ):
        if array.ndim != 4:
            message = (
                'Only 4D images (channels, i, j, k) are supported,'
                f' not {array.shape}'
            )
            raise NotImplementedError(message)
        interpolation_dict = {
            Interpolation.NEAREST: sitk.sitkNearestNeighbor,
            Interpolation.LINEAR: sitk.sitkLinear,
        }
        for i, channel_array in enumerate(array):  # use sitk.VectorImage?
            image = self.nib_to_sitk(channel_array, affine)
            scaling_transform = self.get_scaling_transform(scaling_params)
            rotation_transform = self.get_rotation_transform(rotation_params)
            transform = sitk.Transform(3, sitk.sitkComposite)
            transform.AddTransform(scaling_transform)
            transform.AddTransform(rotation_transform)
            resampled = sitk.Resample(
                image,
                transform,
                interpolation_dict[interpolation],
            )
            channel_array = sitk.GetArrayFromImage(resampled)
            channel_array = channel_array.transpose(2, 1, 0)  # ITK to NumPy
            array[i] = channel_array
        return array

    @staticmethod
    def nib_to_sitk(array, affine):
        """
        TODO: figure out how to get directions from affine
        so that I don't need this
        """
        import nibabel as nib
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(suffix='.nii') as f:
            nib.Nifti1Image(array, affine).to_filename(f.name)
            image = sitk.ReadImage(f.name)
        return image
