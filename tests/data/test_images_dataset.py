#!/usr/bin/env python

"""Tests for ImagesDataset."""

import nibabel as nib
from torchio import DATA, ImagesDataset
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
        with self.assertRaises(TypeError):
            self.iterate_dataset([{}])

    def test_wrong_index(self):
        with self.assertRaises(ValueError):
            self.dataset[:3]

    def test_save_sample(self):
        dataset = ImagesDataset(
            self.subjects_list, transform=lambda x: x)
        _ = len(dataset)  # for coverage
        sample = dataset[0]
        output_path = self.dir / 'test.nii.gz'
        paths_dict = {'t1': output_path}
        dataset.save_sample(sample, paths_dict)
        nii = nib.load(str(output_path))
        ndims_output = len(nii.shape)
        ndims_sample = len(sample['t1'].shape)
        assert ndims_sample == ndims_output + 1

    def test_no_load(self):
        dataset = ImagesDataset(
            self.subjects_list, load_image_data=False)
        for _ in dataset:
            pass

    def test_no_load_transform(self):
        with self.assertRaises(ValueError):
            ImagesDataset(
                self.subjects_list,
                load_image_data=False,
                transform=lambda x: x,
            )

    def test_wrong_transform_init(self):
        with self.assertRaises(ValueError):
            ImagesDataset(
                self.subjects_list,
                transform=dict(),
            )

    def test_wrong_transform_arg(self):
        with self.assertRaises(ValueError):
            self.dataset.set_transform(1)

    @staticmethod
    def iterate_dataset(subjects_list):
        dataset = ImagesDataset(subjects_list)
        for _ in dataset:
            pass
