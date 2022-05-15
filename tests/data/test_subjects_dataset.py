#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader

import torchio as tio
from ..utils import TorchioTestCase


class TestSubjectsDataset(TorchioTestCase):

    def test_indexing_nonint(self):
        dset = tio.SubjectsDataset(self.subjects_list)
        dset[torch.tensor(0)]

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
            tio.SubjectsDataset(
                self.subjects_list,
                transform={},
            )

    def test_wrong_transform_arg(self):
        with self.assertRaises(ValueError):
            self.dataset.set_transform(1)

    @staticmethod
    def iterate_dataset(subjects_list):
        dataset = tio.SubjectsDataset(subjects_list)
        for _ in dataset:
            pass

    def test_from_batch(self):
        dataset = tio.SubjectsDataset([self.sample_subject])
        loader = DataLoader(dataset)
        batch = tio.utils.get_first_item(loader)
        new_dataset = tio.SubjectsDataset.from_batch(batch)
        self.assertTensorEqual(
            dataset[0].t1.data,
            new_dataset[0].t1.data,
        )
