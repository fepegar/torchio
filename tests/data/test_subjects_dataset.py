#!/usr/bin/env python

"""Tests for SubjectsDataset."""

import nibabel as nib
import torchio
from torchio import DATA, SubjectsDataset
from ..utils import TorchioTestCase


class TestSubjectsDataset(TorchioTestCase):
    """Tests for `SubjectsDataset`."""

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
        dataset = SubjectsDataset(
            self.subjects_list, transform=lambda x: x)
        _ = len(dataset)  # for coverage
        sample = dataset[0]
        output_path = self.dir / 'test.nii.gz'
        paths_dict = {'t1': output_path}
        with self.assertWarns(DeprecationWarning):
            dataset.save_sample(sample, paths_dict)
        nii = nib.load(str(output_path))
        ndims_output = len(nii.shape)
        ndims_sample = len(sample['t1'].shape)
        assert ndims_sample == ndims_output + 1

    def test_wrong_transform_init(self):
        with self.assertRaises(ValueError):
            SubjectsDataset(
                self.subjects_list,
                transform=dict(),
            )

    def test_wrong_transform_arg(self):
        with self.assertRaises(ValueError):
            self.dataset.set_transform(1)

    @staticmethod
    def iterate_dataset(subjects_list):
        dataset = SubjectsDataset(subjects_list)
        for _ in dataset:
            pass

    def test_data_loader(self):
        from torch.utils.data import DataLoader
        subj_list = [torchio.datasets.Colin27()]
        dataset = SubjectsDataset(subj_list)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        for batch in loader:
            batch['t1'][DATA]
            batch['brain'][DATA]
