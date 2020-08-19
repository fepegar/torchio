import copy
import shutil
import random
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
import nibabel as nib
from numpy.testing import assert_array_equal, assert_raises
from torchio.datasets import IXITiny
from torchio import DATA, AFFINE, ScalarImage, LabelMap, SubjectsDataset, Subject


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
            t1=ScalarImage(self.get_image_path('t1_a')),
        )
        subject_b = Subject(
            t1=ScalarImage(self.get_image_path('t1_b')),
            label=LabelMap(self.get_image_path('label_b', binary=True)),
        )
        subject_c = Subject(
            label=LabelMap(self.get_image_path('label_c', binary=True)),
        )
        subject_d = Subject(
            t1=ScalarImage(
                self.get_image_path('t1_d'),
                pre_affine=registration_matrix,
            ),
            t2=ScalarImage(self.get_image_path('t2_d')),
            label=LabelMap(self.get_image_path('label_d', binary=True)),
        )
        subject_a4 = Subject(
            t1=ScalarImage(self.get_image_path('t1_a'), components=2),
        )
        self.subjects_list = [
            subject_a,
            subject_a4,
            subject_b,
            subject_c,
            subject_d,
        ]
        self.dataset = SubjectsDataset(self.subjects_list)
        self.sample = self.dataset[-1]  # subject_d

    def make_2d(self, sample):
        sample = copy.deepcopy(sample)
        for image in sample.get_images(intensity_only=False):
            image[DATA] = image[DATA][..., :1]
        return sample

    def make_multichannel(self, sample):
        sample = copy.deepcopy(sample)
        for image in sample.get_images(intensity_only=False):
            image[DATA] = torch.cat(4 * (image[DATA],))
        return sample

    def flip_affine_x(self, sample):
        sample = copy.deepcopy(sample)
        for image in sample.get_images(intensity_only=False):
            image[AFFINE] = np.diag((-1, 1, 1, 1)) @ image[AFFINE]
        return sample

    def get_inconsistent_sample(self):
        """Return a sample containing images of different shape."""
        subject = Subject(
            t1=ScalarImage(self.get_image_path('t1_inc')),
            t2=ScalarImage(
                self.get_image_path('t2_inc', shape=(10, 20, 31))),
            label=LabelMap(
                self.get_image_path(
                    'label_inc',
                    shape=(8, 17, 25),
                    binary=True,
                ),
            ),
            label2=LabelMap(
                self.get_image_path(
                    'label2_inc',
                    shape=(18, 17, 25),
                    binary=True,
                ),
            ),
        )
        return subject

    def get_reference_image_and_path(self):
        """Return a reference image and its path"""
        path = self.get_image_path('ref', shape=(10, 20, 31), spacing=(1, 1, 2))
        image = ScalarImage(path)
        return image, path

    def get_sample_with_partial_volume_label_map(self, components=1):
        """Return a sample with a partial-volume label map."""
        return Subject(
            t1=ScalarImage(
                self.get_image_path('t1_d'),
            ),
            label=LabelMap(
                self.get_image_path(
                    'label_d2', binary=False, components=components
                )
            ),
        )

    def tearDown(self):
        """Tear down test fixtures, if any."""
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
            components=1,
            add_nans=False,
            suffix=None,
            force_binary_foreground=True,
            ):
        shape = (*shape, 1) if len(shape) == 2 else shape
        data = np.random.rand(components, *shape)
        if binary:
            data = (data > 0.5).astype(np.uint8)
            if not data.sum() and force_binary_foreground:
                data[..., 0] = 1
        if add_nans:
            data[:] = np.nan
        affine = np.diag((*spacing, 1))
        if suffix is None:
            suffix = random.choice(('.nii.gz', '.nii', '.nrrd', '.img', '.mnc'))
        path = self.dir / f'{stem}{suffix}'
        if np.random.rand() > 0.5:
            path = str(path)
        image = ScalarImage(tensor=data, affine=affine, check_nans=not add_nans)
        image.save(path)
        return path

    def assertTensorNotEqual(self, *args, **kwargs):
        assert_raises(AssertionError, assert_array_equal, *args, **kwargs)

    def assertTensorEqual(self, *args, **kwargs):
        assert_array_equal(*args, **kwargs)
