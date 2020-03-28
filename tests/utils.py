import shutil
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
import torchio
from torchio.datasets import IXITiny
from torchio import INTENSITY, LABEL, DATA, Image, ImagesDataset, Subject


class TorchioTestCase(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        self.dir = Path(tempfile.gettempdir()) / '.torchio_tests'
        self.dir.mkdir(exist_ok=True)
        random.seed(42)
        np.random.seed(42)

        subject_a = Subject(
            Image('t1', self.get_image_path('t1_a'), INTENSITY),
        )
        subject_b = Subject(
            Image('t1', self.get_image_path('t1_b'), INTENSITY),
            Image('label', self.get_image_path('label_b', binary=True), LABEL),
        )
        subject_c = Subject(
            Image('label', self.get_image_path('label_c', binary=True), LABEL),
        )
        subject_d = Subject(
            Image('t1', self.get_image_path('t1_d'), INTENSITY),
            Image('t2', self.get_image_path('t2_d'), INTENSITY),
            Image('label', self.get_image_path('label_d', binary=True), LABEL),
        )
        self.subjects_list = [
            subject_a,
            subject_b,
            subject_c,
            subject_d,
        ]
        self.dataset = ImagesDataset(self.subjects_list)
        self.sample = self.dataset[-1]

    def get_inconsistent_sample(self):
        """Return a sample containing images of different shape."""
        subject = Subject(
            Image('t1', self.get_image_path('t1_d'), INTENSITY),
            Image('t2', self.get_image_path('t2_d', shape=(10, 20, 31)), INTENSITY),
            Image('label', self.get_image_path('label_d', binary=True), LABEL),
        )
        subjects_list = [subject]
        dataset = ImagesDataset([subject])
        return dataset[0]

    def tearDown(self):
        """Tear down test fixtures, if any."""
        print('Deleting', self.dir)
        shutil.rmtree(self.dir)

    def get_ixi_tiny(self):
        root_dir = Path(tempfile.gettempdir()) / 'torchio' / 'ixi_tiny'
        return IXITiny(root_dir, download=True)

    def get_image_path(self, stem, binary=False, shape=(10, 20, 30)):
        data = np.random.rand(*shape)
        if binary:
            data = (data > 0.5).astype(np.uint8)
        affine = np.eye(4)
        suffix = random.choice(('.nii.gz', '.nii'))
        path = self.dir / f'{stem}{suffix}'
        nib.Nifti1Image(data, affine).to_filename(str(path))
        if np.random.rand() > 0.5:
            path = str(path)
        return path
