import copy
import shutil
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
from torchio.datasets import IXITiny
from torchio import INTENSITY, LABEL, DATA, Image, ImagesDataset, Subject


class TorchioTestCase(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        self.dir = Path(tempfile.gettempdir()) / '.torchio_tests'
        self.dir.mkdir(exist_ok=True)
        random.seed(42)
        np.random.seed(42)

        registration_matrix = np.array([
            [1, 0, 0, 10],
            [0, 1, 0, 0],
            [0, 0, 1.2, 0],
            [0, 0, 0, 1]
        ])

        subject_a = Subject(
            t1=Image(self.get_image_path('t1_a'), INTENSITY),
        )
        subject_b = Subject(
            t1=Image(self.get_image_path('t1_b'), INTENSITY),
            label=Image(self.get_image_path('label_b', binary=True), LABEL),
        )
        subject_c = Subject(
            label=Image(self.get_image_path('label_c', binary=True), LABEL),
        )
        subject_d = Subject(
            t1=Image(
                self.get_image_path('t1_d'),
                INTENSITY,
                pre_affine=registration_matrix,
            ),
            t2=Image(self.get_image_path('t2_d'), INTENSITY),
            label=Image(self.get_image_path('label_d', binary=True), LABEL),
        )
        self.subjects_list = [
            subject_a,
            subject_b,
            subject_c,
            subject_d,
        ]
        self.dataset = ImagesDataset(self.subjects_list)
        self.sample = self.dataset[-1]

    def make_2d(self, sample):
        sample = copy.deepcopy(sample)
        for image in sample.get_images(intensity_only=False):
            image[DATA] = image[DATA][:, 0:1, ...]
        return sample

    def get_inconsistent_sample(self):
        """Return a sample containing images of different shape."""
        subject = Subject(
            t1=Image(self.get_image_path('t1_d'), INTENSITY),
            t2=Image(
                self.get_image_path('t2_d', shape=(10, 20, 31)), INTENSITY),
            label=Image(
                self.get_image_path(
                    'label_d',
                    shape=(8, 17, 25),
                    binary=True,
                ),
                LABEL,
            ),
        )
        subjects_list = [subject]
        dataset = ImagesDataset(subjects_list)
        return dataset[0]

    def get_reference_image_and_path(self):
        """Return a reference image and its path"""
        path = self.get_image_path('ref', shape=(10, 20, 31))
        image = Image(path, INTENSITY)
        return image, path

    def tearDown(self):
        """Tear down test fixtures, if any."""
        print('Deleting', self.dir)
        shutil.rmtree(self.dir)

    def get_ixi_tiny(self):
        root_dir = Path(tempfile.gettempdir()) / 'torchio' / 'ixi_tiny'
        return IXITiny(root_dir, download=True)

    def get_image_path(
            self,
            stem,
            binary=False,
            shape=(10, 20, 30),
            spacing=(1, 1, 1),
            ):
        data = np.random.rand(*shape)
        if binary:
            data = (data > 0.5).astype(np.uint8)
        affine = np.diag((*spacing, 1))
        suffix = random.choice(('.nii.gz', '.nii'))
        path = self.dir / f'{stem}{suffix}'
        nib.Nifti1Image(data, affine).to_filename(str(path))
        if np.random.rand() > 0.5:
            path = str(path)
        return path
