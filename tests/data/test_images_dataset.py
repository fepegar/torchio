#!/usr/bin/env python

"""Tests for ImagesDataset."""

import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
import torchio
from torchio import INTENSITY, LABEL, DATA, Image
from ..utils import TorchioTestCase


class TestImagesDataset(TorchioTestCase):
    """Tests for `ImagesDataset`."""

    def test_images(self):
        self.iterate_dataset(self.subjects_list)

    def test_empty_subjects_list(self):
        with self.assertRaises(ValueError):
            self.iterate_dataset([])

    def test_empty_subjects_tuple(self):
        with self.assertRaises(ValueError):
            self.iterate_dataset(())

    def test_wrong_subjects_type(self):
        with self.assertRaises(TypeError):
            self.iterate_dataset(0)

    def test_wrong_subject_type_int(self):
        with self.assertRaises(TypeError):
            self.iterate_dataset([0])

    def test_wrong_subject_type_dict(self):
        with self.assertRaises(ValueError):
            self.iterate_dataset([{}])

    def test_image_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.iterate_dataset([[Image('t1', 'nopath', INTENSITY)]])

    def test_wrong_path_type(self):
        with self.assertRaises(TypeError):
            self.iterate_dataset([[Image('t1', 5, INTENSITY)]])

    def test_duplicate_image_name(self):
        with self.assertRaises(KeyError):
            with tempfile.NamedTemporaryFile() as f:
                images = [
                    Image('t1', f.name, INTENSITY),
                    Image('t1', f.name, INTENSITY),
                ]
            self.iterate_dataset([images])

    def test_wrong_image_extension(self):
        with self.assertRaises(RuntimeError):
            path = self.dir / 'test.txt'
            path.touch()
            self.iterate_dataset([[Image('t1', path, INTENSITY)]])

    def test_wrong_index(self):
        with self.assertRaises(TypeError):
            self.dataset[:3]

    def test_coverage(self):
        dataset = torchio.ImagesDataset(
            self.subjects_list, transform=lambda x: x)
        _ = len(dataset)  # for coverage
        sample = dataset[0]
        output_path = self.dir / 'test.nii.gz'
        paths_dict = {'t1': output_path}
        dataset.save_sample(sample, paths_dict)
        nii = nib.load(str(output_path))
        ndims_output = len(nii.shape)
        ndims_sample = len(sample['t1'][DATA].shape)
        assert ndims_sample == ndims_output + 1

    @staticmethod
    def iterate_dataset(subjects_list):
        dataset = torchio.ImagesDataset(subjects_list)
        for _ in dataset:
            pass
