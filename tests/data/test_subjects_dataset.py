#!/usr/bin/env python

from torchio import DATA, SubjectsDataset
from ..utils import TorchioTestCase


class TestSubjectsDataset(TorchioTestCase):

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

    def test_wrong_transform_init(self):
        with self.assertRaises(ValueError):
            SubjectsDataset(
                self.subjects_list,
                transform={},
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
        subj_list = [self.sample_subject]
        dataset = SubjectsDataset(subj_list)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        for batch in loader:
            batch['t1'][DATA]
            batch['label'][DATA]
