import copy

import pytest
import torch

import torchio as tio

from .utils import TorchioTestCase


class TestUtils(TorchioTestCase):
    """Tests for `utils` module."""

    def test_to_tuple(self):
        assert tio.utils.to_tuple(1) == (1,)
        assert tio.utils.to_tuple((1,)) == (1,)
        assert tio.utils.to_tuple(1, length=3) == (1, 1, 1)
        assert tio.utils.to_tuple((1, 2)) == (1, 2)
        assert tio.utils.to_tuple((1, 2), length=3) == (1, 2)
        assert tio.utils.to_tuple([1, 2], length=3) == (1, 2)

    def test_get_stem(self):
        assert tio.utils.get_stem('/home/image.nii.gz') == 'image'
        assert tio.utils.get_stem('/home/image.nii') == 'image'
        assert tio.utils.get_stem('/home/image.nrrd') == 'image'

    def test_guess_type(self):
        assert tio.utils.guess_type('None') is None
        assert isinstance(tio.utils.guess_type('1'), int)
        assert isinstance(tio.utils.guess_type('1.5'), float)
        assert isinstance(tio.utils.guess_type('(1, 3, 5)'), tuple)
        assert isinstance(tio.utils.guess_type('(1,3,5)'), tuple)
        assert isinstance(tio.utils.guess_type('[1,3,5]'), list)
        assert isinstance(tio.utils.guess_type('test'), str)

    def test_apply_transform_to_file(self):
        transform = tio.RandomFlip()
        tio.utils.apply_transform_to_file(
            self.get_image_path('input'),
            transform,
            self.get_image_path('output'),
            verbose=True,
        )

    def test_subjects_from_batch(self):
        dataset = tio.SubjectsDataset(4 * [self.sample_subject])
        loader = tio.SubjectsLoader(dataset, batch_size=4)
        batch = tio.utils.get_first_item(loader)
        subjects = tio.utils.get_subjects_from_batch(batch)
        assert isinstance(subjects[0], tio.Subject)

    def test_subjects_from_batch_with_string_metadata(self):
        subject_c_with_string_metadata = tio.Subject(
            name='John Doe',
            label=tio.LabelMap(self.get_image_path('label_c', binary=True)),
        )

        dataset = tio.SubjectsDataset(4 * [subject_c_with_string_metadata])
        loader = tio.SubjectsLoader(dataset, batch_size=4)
        batch = tio.utils.get_first_item(loader)
        subjects = tio.utils.get_subjects_from_batch(batch)
        assert isinstance(subjects[0], tio.Subject)
        assert 'label' in subjects[0]
        assert 'name' in subjects[0]

    def test_subjects_from_batch_with_int_metadata(self):
        subject_c_with_int_metadata = tio.Subject(
            age=45,
            label=tio.LabelMap(self.get_image_path('label_c', binary=True)),
        )
        dataset = tio.SubjectsDataset(4 * [subject_c_with_int_metadata])
        loader = tio.SubjectsLoader(dataset, batch_size=4)
        batch = tio.utils.get_first_item(loader)
        subjects = tio.utils.get_subjects_from_batch(batch)
        assert isinstance(subjects[0], tio.Subject)
        assert 'label' in subjects[0]
        assert 'age' in subjects[0]

    def test_add_images_from_batch(self):
        subject = copy.deepcopy(self.sample_subject)
        subjects = 4 * [subject]
        preds = torch.rand(4, *subject.shape)
        tio.utils.add_images_from_batch(subjects, preds)

    def test_empty_batch(self):
        with pytest.raises(RuntimeError):
            tio.utils.get_batch_images_and_size({})
