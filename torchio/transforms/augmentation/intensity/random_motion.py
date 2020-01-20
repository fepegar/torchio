"""
Custom implementation of

    Shaw et al., 2019
    MRI k-Space Motion Artefact Augmentation:
    Model Robustness and Task-Specific Uncertainty

"""

import warnings
import torch
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from scipy.linalg import logm, expm
from ....utils import is_image_dict
from ....torchio import INTENSITY, DATA, AFFINE
from .. import Interpolation
from .. import RandomTransform


class RandomMotion(RandomTransform):
    def __init__(
            self,
            degrees=10,
            translation=10,  # in mm
            num_transforms=2,
            image_interpolation=Interpolation.LINEAR,
            proportion_to_augment=0.5,
            seed=None,
            verbose=False,
            ):
        super().__init__(seed=seed, verbose=verbose)
        self.degrees_range = self.parse_degrees(degrees)
        self.translation_range = self.parse_translation(translation)
        self.num_transforms = num_transforms
        self.image_interpolation = image_interpolation
        self.proportion_to_augment = self.parse_probability(
            proportion_to_augment,
            'proportion_to_augment',
        )

    def apply_transform(self, sample):
        for image_name, image_dict in sample.items():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] != INTENSITY:
                continue
            params = self.get_params(
                self.degrees_range,
                self.translation_range,
                self.num_transforms,
                self.proportion_to_augment
            )
            times_params, degrees_params, translation_params, do_it = params
            keys = (
                'random_motion_times',
                'random_motion_degrees',
                'random_motion_translation',
                'random_motion_do',
            )
            for key, p in zip(keys, params):
                sample[image_name][key] = p
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
            transforms = self.get_rigid_transforms(
                degrees_params,
                translation_params,
                image,
            )
            transforms = self.demean_transforms(
                transforms,
                times_params,
            )
            image_dict[DATA] = self.add_artifact(
                image,
                transforms,
                times_params,
                self.image_interpolation,
            )
            # Add channels dimension
            image_dict[DATA] = image_dict[DATA][np.newaxis, ...]
            image_dict[DATA] = torch.from_numpy(image_dict[DATA])
        return sample

    @staticmethod
    def get_params(
            degrees_range,
            translation_range,
            num_transforms,
            probability,
            perturbation=0.3,
            ):
        """
        If perturbation is 0, the intervals between movements are constant
        """
        degrees_params = get_params_array(
            degrees_range, num_transforms)
        translation_params = get_params_array(
            translation_range, num_transforms)
        step = 1 / (num_transforms + 1)
        times = torch.arange(0, 1, step)[1:]
        noise = torch.FloatTensor(num_transforms)
        noise.uniform_(-step * perturbation, step * perturbation)
        times += noise
        times_params = times.numpy()
        do_it = torch.rand(1) < probability
        return times_params, degrees_params, translation_params, do_it

    def get_rigid_transforms(self, degrees_params, translation_params, image):
        center_ijk = np.array(image.GetSize()) / 2
        center_lps = image.TransformContinuousIndexToPhysicalPoint(center_ijk)
        identity = np.eye(4)
        matrices = [identity]
        for degrees, translation in zip(degrees_params, translation_params):
            radians = np.radians(degrees).tolist()
            motion = sitk.Euler3DTransform()
            motion.SetCenter(center_lps)
            motion.SetRotation(*radians)
            motion.SetTranslation(translation.tolist())
            motion_matrix = self.transform_to_matrix(motion)
            matrices.append(motion_matrix)
        transforms = [self.matrix_to_transform(m) for m in matrices]
        return transforms

    def demean_transforms(self, transforms, times):
        matrices = [self.transform_to_matrix(t) for t in transforms]
        times = np.insert(times, 0, 0)
        times = np.append(times, 1)
        weights = np.diff(times)
        mean = self.matrix_average(matrices, weights=weights)
        inverse_mean = np.linalg.inv(mean)
        demeaned_matrices = [inverse_mean @ matrix for matrix in matrices]
        demeaned_transforms = [
            self.matrix_to_transform(m) for m in demeaned_matrices]
        return demeaned_transforms

    @staticmethod
    def transform_to_matrix(transform):
        matrix = np.eye(4)
        rotation = np.array(transform.GetMatrix()).reshape(3, 3)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = transform.GetTranslation()
        return matrix

    @staticmethod
    def matrix_to_transform(matrix):
        transform = sitk.Euler3DTransform()
        rotation = matrix[:3, :3].flatten().tolist()
        transform.SetMatrix(rotation)
        transform.SetTranslation(matrix[:3, 3])
        return transform

    def resample_images(self, image, transforms, interpolation):
        floating = reference = image
        default_value = np.float64(sitk.GetArrayViewFromImage(image).min())
        transforms = transforms[1:]  # first is identity
        images = [image]  # first is identity
        trsfs = tqdm(transforms, leave=False) if self.verbose else transforms
        for transform in trsfs:
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(interpolation.value)
            resampler.SetReferenceImage(reference)
            resampler.SetOutputPixelType(sitk.sitkFloat32)
            resampler.SetDefaultPixelValue(default_value)
            resampler.SetTransform(transform)
            resampled = resampler.Execute(floating)
            images.append(resampled)
        return images

    @staticmethod
    def sort_spectra(spectra, times):
        """
        Use original spectrum to fill the center of K-space
        """
        num_spectra = len(spectra)
        if np.any(times > 0.5):
            index = np.where(times > 0.5)[0].min()
        else:
            index = num_spectra - 1
        spectra[0], spectra[index] = spectra[index], spectra[0]

    def add_artifact(
            self,
            image,
            transforms,
            times,
            interpolation: Interpolation,
            ):
        images = self.resample_images(image, transforms, interpolation)
        arrays = [sitk.GetArrayViewFromImage(im) for im in images]
        arrays = [array.transpose() for array in arrays]  # ITK to NumPy
        arrays = tqdm(arrays, leave=False) if self.verbose else arrays
        spectra = [self.fourier_transform(array) for array in arrays]
        self.sort_spectra(spectra, times)
        result_spectrum = np.empty_like(spectra[0])
        last_index = result_spectrum.shape[2]
        indices = (last_index * times).astype(int).tolist()
        indices.append(last_index)
        ini = 0
        for spectrum, fin in zip(spectra, indices):
            result_spectrum[..., ini:fin] = spectrum[..., ini:fin]
            ini = fin
        result_image = self.inv_fourier_transform(result_spectrum)
        return result_image.astype(np.float32)

    @staticmethod
    def fourier_transform(array):
        transformed = np.fft.fft2(array)
        fshift = np.fft.fftshift(transformed)
        return fshift

    @staticmethod
    def inv_fourier_transform(fshift):
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        return np.abs(img_back)

    def matrix_average(self, matrices, weights=None):
        if weights is None:
            num_matrices = len(matrices)
            weights = num_matrices * (1 / num_matrices,)
        logs = [w * logm(A) for (w, A) in zip(weights, matrices)]
        logs = np.array(logs)
        logs_sum = logs.sum(axis=0)
        return expm(logs_sum)


def get_params_array(nums_range, num_transforms):
    tensor = torch.FloatTensor(num_transforms, 3).uniform_(*nums_range)
    return tensor.numpy()
