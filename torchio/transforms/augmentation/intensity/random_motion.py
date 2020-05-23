"""
Custom implementation of

    Shaw et al., 2019
    MRI k-Space Motion Artefact Augmentation:
    Model Robustness and Task-Specific Uncertainty

"""

import warnings
from typing import Tuple, Optional, List
import torch
import numpy as np
import SimpleITK as sitk
from ....torchio import DATA, AFFINE
from ....data.subject import Subject
from .. import Interpolation, get_sitk_interpolator
from .. import RandomTransform


class RandomMotion(RandomTransform):
    r"""Add random MRI motion artifact.

    Custom implementation of `Shaw et al. 2019, MRI k-Space Motion Artefact
    Augmentation: Model Robustness and Task-Specific
    Uncertainty <http://proceedings.mlr.press/v102/shaw19a.html>`_.

    Args:
        degrees: Tuple :math:`(a, b)` defining the rotation range in degrees of
            the simulated movements. The rotation angles around each axis are
            :math:`(\theta_1, \theta_2, \theta_3)`,
            where :math:`\theta_i \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`\theta_i \sim \mathcal{U}(-d, d)`.
            Larger values generate more distorted images.
        translation: Tuple :math:`(a, b)` defining the translation in mm of
            the simulated movements. The translations along each axis are
            :math:`(t_1, t_2, t_3)`,
            where :math:`t_i \sim \mathcal{U}(a, b)`.
            If only one value :math:`t` is provided,
            :math:`t_i \sim \mathcal{U}(-t, t)`.
            Larger values generate more distorted images.
        num_transforms: Number of simulated movements.
            Larger values generate more distorted images.
        image_interpolation: See :ref:`Interpolation`.
        p: Probability that this transform will be applied.
        seed: See :py:class:`~torchio.transforms.augmentation.RandomTransform`.

    .. warning:: Large numbers of movements lead to longer execution times for
        3D images.
    """
    def __init__(
            self,
            degrees: float = 10,
            translation: float = 10,  # in mm
            num_transforms: int = 2,
            image_interpolation: str = 'linear',
            p: float = 1,
            seed: Optional[int] = None,
            ):
        super().__init__(p=p, seed=seed)
        self.degrees_range = self.parse_degrees(degrees)
        self.translation_range = self.parse_translation(translation)
        self.num_transforms = num_transforms
        self.image_interpolation = self.parse_interpolation(image_interpolation)

    def apply_transform(self, sample: Subject) -> dict:
        random_parameters_images_dict = {}
        for image_name, image_dict in sample.get_images_dict().items():
            data = image_dict[DATA]
            is_2d = data.shape[-3] == 1
            params = self.get_params(
                self.degrees_range,
                self.translation_range,
                self.num_transforms,
                is_2d=is_2d,
            )
            times_params, degrees_params, translation_params = params
            random_parameters_dict = {
                'times': times_params,
                'degrees': degrees_params,
                'translation': translation_params,
            }
            random_parameters_images_dict[image_name] = random_parameters_dict
            if (data[0] < -0.1).any():
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
                data[0],
                image_dict[AFFINE],
            )
            transforms = self.get_rigid_transforms(
                degrees_params,
                translation_params,
                image,
            )
            data = self.add_artifact(
                image,
                transforms,
                times_params,
                self.image_interpolation,
            )
            # Add channels dimension
            data = data[np.newaxis, ...]
            image_dict[DATA] = torch.from_numpy(data)
        sample.add_transform(self, random_parameters_images_dict)
        return sample

    @staticmethod
    def get_params(
            degrees_range: Tuple[float, float],
            translation_range: Tuple[float, float],
            num_transforms: int,
            perturbation: float = 0.3,
            is_2d: bool = False,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        # If perturbation is 0, time intervals between movements are constant
        degrees_params = get_params_array(
            degrees_range, num_transforms)
        translation_params = get_params_array(
            translation_range, num_transforms)
        if is_2d:  # imagine sagittal (1, A, S)
            degrees_params[:, -2:] = 0  # rotate around R axis only
            translation_params[:, 0] = 0  # translate in AS plane only
        step = 1 / (num_transforms + 1)
        times = torch.arange(0, 1, step)[1:]
        noise = torch.FloatTensor(num_transforms)
        noise.uniform_(-step * perturbation, step * perturbation)
        times += noise
        times_params = times.numpy()
        return times_params, degrees_params, translation_params

    def get_rigid_transforms(
            self,
            degrees_params: np.ndarray,
            translation_params: np.ndarray,
            image: sitk.Image,
            ) -> List[sitk.Euler3DTransform]:
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

    @staticmethod
    def transform_to_matrix(transform: sitk.Euler3DTransform) -> np.ndarray:
        matrix = np.eye(4)
        rotation = np.array(transform.GetMatrix()).reshape(3, 3)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = transform.GetTranslation()
        return matrix

    @staticmethod
    def matrix_to_transform(matrix: np.ndarray) -> sitk.Euler3DTransform:
        transform = sitk.Euler3DTransform()
        rotation = matrix[:3, :3].flatten().tolist()
        transform.SetMatrix(rotation)
        transform.SetTranslation(matrix[:3, 3])
        return transform

    @staticmethod
    def resample_images(
            image: sitk.Image,
            transforms: List[sitk.Euler3DTransform],
            interpolation: Interpolation,
            ) -> List[sitk.Image]:
        floating = reference = image
        default_value = np.float64(sitk.GetArrayViewFromImage(image).min())
        transforms = transforms[1:]  # first is identity
        images = [image]  # first is identity
        for transform in transforms:
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(get_sitk_interpolator(interpolation))
            resampler.SetReferenceImage(reference)
            resampler.SetOutputPixelType(sitk.sitkFloat32)
            resampler.SetDefaultPixelValue(default_value)
            resampler.SetTransform(transform)
            resampled = resampler.Execute(floating)
            images.append(resampled)
        return images

    @staticmethod
    def sort_spectra(spectra: np.ndarray, times: np.ndarray):
        """Use original spectrum to fill the center of k-space"""
        num_spectra = len(spectra)
        if np.any(times > 0.5):
            index = np.where(times > 0.5)[0].min()
        else:
            index = num_spectra - 1
        spectra[0], spectra[index] = spectra[index], spectra[0]

    def add_artifact(
            self,
            image: sitk.Image,
            transforms: List[sitk.Euler3DTransform],
            times: np.ndarray,
            interpolation: Interpolation,
            ):
        images = self.resample_images(image, transforms, interpolation)
        arrays = [sitk.GetArrayViewFromImage(im) for im in images]
        arrays = [array.transpose() for array in arrays]  # ITK to NumPy
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


def get_params_array(nums_range: Tuple[float, float], num_transforms: int):
    tensor = torch.FloatTensor(num_transforms, 3).uniform_(*nums_range)
    return tensor.numpy()
