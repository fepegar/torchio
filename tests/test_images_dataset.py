#!/usr/bin/env python

"""Tests for ImagesDataset."""

import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
import torchio
from torchio import INTENSITY, LABEL, Image


class TestRandomElasticDeformation(unittest.TestCase):
    """Tests for `RandomElasticDeformation`."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.dir = Path(tempfile.gettempdir()) / '.torchio_tests'
        self.dir.mkdir(exist_ok=True)
        random.seed(42)
        np.random.seed(42)

        subject_a = [
            Image('t1', self.get_image_path('t1_a'), INTENSITY),
        ]
        subject_b = [
            Image('t1', self.get_image_path('t1_b'), INTENSITY),
            Image('label', self.get_image_path('label_b'), LABEL),
        ]
        subject_c = [
            Image('label', self.get_image_path('label_c'), LABEL),
        ]
        subject_d = [
            Image('t1', self.get_image_path('t1_d'), INTENSITY),
            Image('t2', self.get_image_path('t2_d'), INTENSITY),
            Image('label', self.get_image_path('label_d'), LABEL),
        ]
        self.subjects_list = [
            subject_a,
            subject_b,
            subject_c,
            subject_d,
        ]

    def tearDown(self):
        """Tear down test fixtures, if any."""
        import shutil
        shutil.rmtree(self.dir)

    def get_image_path(self, stem):
        data = np.random.rand(10, 20, 30)
        affine = np.eye(4)
        suffix = random.choice(('.nii.gz', '.nii'))
        path = self.dir / f'{stem}{suffix}'
        nib.Nifti1Image(data, affine).to_filename(str(path))
        path = str(path) if np.random.rand() > 0.5 else path
        return path

    def test_images(self):
        self.iterate_dataset(self.subjects_list)

    def test_wrong_subjects_list(self):
        with self.assertRaises(ValueError):
            self.iterate_dataset([])
        with self.assertRaises(ValueError):
            self.iterate_dataset(())
        with self.assertRaises(TypeError):
            self.iterate_dataset(0)
        with self.assertRaises(TypeError):
            self.iterate_dataset([0])
        with self.assertRaises(TypeError):
            self.iterate_dataset([{}])
        with self.assertRaises(FileNotFoundError):
            self.iterate_dataset([[Image('t1', 'nopath', INTENSITY)]])
        with self.assertRaises(TypeError):
            self.iterate_dataset([[Image('t1', 5, INTENSITY)]])
        with self.assertRaises(KeyError):
            with tempfile.NamedTemporaryFile() as f:
                images = [
                    Image('t1', f.name, INTENSITY),
                    Image('t1', f.name, INTENSITY),
                ]
            self.iterate_dataset([images])
        with self.assertRaises(ValueError):
            path = self.dir / 'test.txt'
            path.touch()
            self.iterate_dataset([[Image('t1', path, INTENSITY)]])

    def test_others(self):
        dataset = torchio.ImagesDataset(
            self.subjects_list, verbose=True, transform=lambda x: x)
        _ = len(dataset)  # for coverage
        sample = dataset[0]
        output_path = self.dir / 'test.nii.gz'
        paths_dict = {'t1': output_path}
        dataset.save_sample(sample, paths_dict)
        nii = nib.load(str(output_path))
        ndims_output = len(nii.shape)
        ndims_sample = len(sample['t1']['data'].shape)
        assert ndims_sample == ndims_output + 1

    def iterate_dataset(self, subjects_list):
        dataset = torchio.ImagesDataset(subjects_list)
        for _ in dataset:
            pass
