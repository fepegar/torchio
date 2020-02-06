import warnings
import torch
import numpy as np
import nibabel as nb
from nibabel.processing import resample_from_to
from tqdm import tqdm
import SimpleITK as sitk
from scipy.linalg import logm, expm
from ....utils import is_image_dict
from ....torchio import INTENSITY, DATA, AFFINE
from .. import Interpolation
from .. import RandomTransform


class RandomMotionTimeCourseAffines(RandomTransform):

    def __init__(self, fitpars, time_points=[0.0, 0.25, 0.5, 0.75], pct_oversampling=0.0, combine_axis=2,
                 seed=None, verbose=False):
        super().__init__(seed=seed, verbose=verbose)
        self.fitpars = fitpars
        self.time_points = time_points
        self.pct_oversampling = pct_oversampling
        self.affines = self.extract_affines_from_timecourse()
        self.combine_axis = combine_axis

    def combine_kspaces(self, spectra, axis=2):
        """
        Combines the k_spaces in spectra according to times values, if not specified, uniform times are assumed
        """
        nb_spectrums = len(spectra)
        spects = spectra
        kspace_shape = spects[0].shape

        result_spectrum = np.empty_like(spects[0])
        times = np.linspace(0, 1, nb_spectrums + 1, endpoint=True)[1:] if self.time_points is None else np.asarray(self.time_points)
        indices = (kspace_shape[axis] * times).astype(int) + 1
        start_idx = 0
        slicing = [slice(None)] * 3
        for spectrum, fin in zip(spects, indices):
            slicing[axis] = slice(start_idx, fin)
            result_spectrum[slicing] = spectrum[slicing]
            start_idx = fin
        result_image = abs(self._ifft_im(result_spectrum))
        return result_image.astype(np.float32)

    @staticmethod
    def fitpars_to_matrix(fitpars):
        '''
        Converts fitpars (6 params : 3 translations and 3 rotations) to the corresponding affine transform matrix (4x4)
        '''
        transform = sitk.Euler3DTransform()
        transform.SetTranslation(fitpars[:3])
        transform.SetRotation(*np.radians(fitpars[3:]))
        matrix = np.empty((4, 4))
        rotation = np.array(transform.GetMatrix()).reshape(3, 3)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = transform.GetTranslation()
        return matrix

    def extract_affines_from_timecourse(self):
        '''
        Extract the affines found at the specified timepoints in the fitpars timecourse
        '''
        fpars_len = self.fitpars.shape[1]
        affines = []
        for time in self.time_points:
            extraction_index = int(fpars_len*time) - 1
            fpars = self.fitpars[:, extraction_index]
            affines.append(self.fitpars_to_matrix(fpars))
        return affines

    def apply_affines(self, data, affines, orig_affine=np.eye(4)):
        trsfms_affine = [orig_affine.dot(trsfm_affine) for trsfm_affine in affines]
        nb_data = data
        if isinstance(nb_data, np.ndarray):
            nb_data = nb.Nifti1Image(nb_data, orig_affine)
        resampled_data = [resample_from_to(nb_data, (data.shape, trsfm_affine)) for trsfm_affine in trsfms_affine]
        return resampled_data

    def apply_transform(self, sample):
        for image_name, image_dict in sample.items():
            if not is_image_dict(image_dict):
                continue
            if image_dict['type'] != INTENSITY:
                continue

            data = np.squeeze(image_dict["data"])
            original_shape = data.shape
            oversampled_data = self._oversample(data, self.pct_oversampling)
            transformed_data = self.apply_affines(oversampled_data, self.affines, orig_affine=np.eye(4))
            transformed_kspace_data = [self._fft_im(corrupted_data.get_fdata()) for corrupted_data in transformed_data]
            artifacted_data = self.combine_kspaces(transformed_kspace_data, axis=self.combine_axis)
            cropped_artifacted_data = self.crop_volume(artifacted_data, original_shape)

            image_dict["data"] = torch.from_numpy(cropped_artifacted_data[np.newaxis, ...])
        return sample

    @staticmethod
    def get_params(*args, **kwargs):
        pass
