"""
Simplified implementation of

    Shaw et al., 2019
    MRI k-Space Motion Artefact Augmentation:
    Model Robustness and Task-Specific Uncertainty

Matrix algebra functions from

    Alexa, 2002
    Linear combination of transformations
"""

import torch
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from ..torchio import LABEL
from ..utils import is_image_dict
from .interpolation import Interpolation
from .random_transform import RandomTransform


class RandomMotion(RandomTransform):
    def __init__(
            self,
            degrees=30,
            translation=10,  # in mm
            num_transforms=2,
            image_interpolation=Interpolation.LINEAR,
            seed=None,
            verbose=False,
            ):
        super().__init__(seed=seed, verbose=verbose)
        self.degrees_range = self.parse_degrees(degrees)
        self.translation_range = self.parse_translation(translation)
        self.num_transforms = num_transforms
        self.image_interpolation = image_interpolation

    def apply_transform(self, sample):
        for image_name, image_dict in sample.items():
            times_params, degrees_params, translation_params = self.get_params(
                self.degrees_range,
                self.translation_range,
                self.num_transforms,
            )
            keys = (
                'random_motion_times',
                'random_motion_degrees',
                'random_motion_translation',
            )
            all_params = times_params, degrees_params, translation_params
            for key, params in zip(keys, all_params):
                sample[image_name][key] = params
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] == LABEL:
                interpolation = Interpolation.NEAREST
            else:
                interpolation = self.image_interpolation
            image = self.nib_to_sitk(
                image_dict['data'].squeeze(),
                image_dict['affine'],
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
            image_dict['data'] = self.add_artifact(
                image,
                transforms,
                times_params,
                interpolation,
            )
            # Add channels dimension
            image_dict['data'] = image_dict['data'][np.newaxis, ...]
        return sample

    @staticmethod
    def get_params(degrees_range, translation_range, num_transforms):
        degrees_params = get_params_array(
            degrees_range, num_transforms)
        translation_params = get_params_array(
            translation_range, num_transforms)
        times_params = np.array(sorted(torch.rand(num_transforms).tolist()))
        return times_params, degrees_params, translation_params

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
        interpolator = interpolation.value
        transforms = transforms[1:]  # first is identity
        images = [image]  # first is identity
        trsfs = tqdm(transforms, leave=False) if self.verbose else transforms
        for transform in trsfs:
            resampled = sitk.Resample(
                floating,
                reference,
                transform,
                interpolator,
            )
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
        arrays = [array.transpose(2, 1, 0) for array in arrays]  # ITK to NumPy
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

    # The following methods are from (Alexa, 2002)
    @staticmethod
    def matrix_sqrt(A, epsilon=1e-9):
        X = A.copy()
        Y = np.eye(4)
        diff = np.inf
        while diff > epsilon:
            iX = np.linalg.inv(X)
            iY = np.linalg.inv(Y)
            X = 1 / 2 * (X + iY)
            Y = 1 / 2 * (Y + iX)
            diff = np.linalg.norm(X @ X - A)
        return X

    @staticmethod
    def matrix_exp(A, q=6):
        identity = np.eye(4, dtype=float)
        norm_a = np.linalg.norm(A)
        log = np.log2(norm_a)
        j = int(max(0, 1 + np.floor(log))) if norm_a > 0 else 0
        A = 2 ** (-j) * A
        D = identity.copy()
        N = identity.copy()
        X = identity.copy()
        c = 1
        for k in range(1, q + 1):
            c = c * (q - k + 1) / (k * (2 * q - k + 1))
            X = A @ X
            N += c * X
            D += (-1) ** k * c * X
        X = np.linalg.inv(D) @ N
        X = np.linalg.matrix_power(X, 2 * j)
        return X

    def matrix_log(self, A, epsilon=1e-9):
        identity = np.eye(4)
        k = 0
        diff = np.inf
        while diff > 0.5:
            A = self.matrix_sqrt(A)
            k += 1
            diff = np.linalg.norm(A - identity)
        A = identity - A
        Z = A.copy()
        X = A.copy()
        i = 1
        while np.linalg.norm(Z) > epsilon:
            Z = Z @ A
            i += 1
            X += Z / i
        X = 2 ** k * X
        return X

    def matrix_average(self, matrices, weights=None):
        if weights is None:
            num_matrices = len(matrices)
            weights = num_matrices * (1 / num_matrices,)
        logs = [w * self.matrix_log(A) for (w, A) in zip(weights, matrices)]
        logs = np.array(logs)
        logs_sum = logs.sum(axis=0)
        return self.matrix_exp(logs_sum)


def get_params_array(nums_range, num_transforms):
    # TODO: sample from Poisson distribution? Gaussian?
    tensor = torch.FloatTensor(num_transforms, 3).uniform_(*nums_range)
    return tensor.numpy()
