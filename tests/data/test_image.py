#!/usr/bin/env python

"""Tests for Image."""

import copy
import torch
import pytest
import numpy as np
from torchio import ScalarImage, LabelMap, Subject, INTENSITY, LABEL, STEM
from ..utils import TorchioTestCase
from torchio import RandomFlip, RandomAffine


class TestImage(TorchioTestCase):
    """Tests for `Image`."""

    def test_image_not_found(self):
        with self.assertRaises(FileNotFoundError):
            ScalarImage('nopath')

    def test_wrong_path_value(self):
        with self.assertRaises(RuntimeError):
            ScalarImage('~&./@#"!?X7=+')

    def test_wrong_path_type(self):
        with self.assertRaises(TypeError):
            ScalarImage(5)

    def test_wrong_affine(self):
        with self.assertRaises(TypeError):
            ScalarImage(5, affine=1)

    def test_tensor_flip(self):
        sample_input = torch.ones((4, 30, 30, 30))
        RandomFlip()(sample_input)

    def test_tensor_affine(self):
        sample_input = torch.ones((4, 10, 10, 10))
        RandomAffine()(sample_input)

    def test_crop_attributes(self):
        cropped = self.sample.crop((1, 1, 1), (5, 5, 5))
        self.assertIs(self.sample.t1['pre_affine'], cropped.t1['pre_affine'])

    def test_crop_does_not_create_wrong_path(self):
        data = torch.ones((1, 10, 10, 10))
        image = ScalarImage(tensor=data)
        cropped = image.crop((1, 1, 1), (5, 5, 5))
        self.assertIs(cropped.path, None)

    def test_scalar_image_type(self):
        data = torch.ones((1, 10, 10, 10))
        image = ScalarImage(tensor=data)
        self.assertIs(image.type, INTENSITY)

    def test_label_map_type(self):
        data = torch.ones((1, 10, 10, 10))
        label = LabelMap(tensor=data)
        self.assertIs(label.type, LABEL)

    def test_wrong_scalar_image_type(self):
        data = torch.ones((1, 10, 10, 10))
        with self.assertRaises(ValueError):
            ScalarImage(tensor=data, type=LABEL)

    def test_wrong_label_map_type(self):
        data = torch.ones((1, 10, 10, 10))
        with self.assertRaises(ValueError):
            LabelMap(tensor=data, type=INTENSITY)

    def test_crop_scalar_image_type(self):
        data = torch.ones((1, 10, 10, 10))
        image = ScalarImage(tensor=data)
        cropped = image.crop((1, 1, 1), (5, 5, 5))
        self.assertIs(cropped.type, INTENSITY)

    def test_crop_label_map_type(self):
        data = torch.ones((1, 10, 10, 10))
        label = LabelMap(tensor=data)
        cropped = label.crop((1, 1, 1), (5, 5, 5))
        self.assertIs(cropped.type, LABEL)

    def test_no_input(self):
        with self.assertRaises(ValueError):
            ScalarImage()

    def test_bad_key(self):
        with self.assertRaises(ValueError):
            ScalarImage(path='', data=5)

    def test_repr(self):
        sample = Subject(t1=ScalarImage(self.get_image_path('repr_test')))
        assert 'shape' not in repr(sample['t1'])
        sample.load()
        assert 'shape' in repr(sample['t1'])

    def test_data_tensor(self):
        sample = copy.deepcopy(self.sample)
        sample.load()
        self.assertIs(sample.t1.data, sample.t1.tensor)

    def test_bad_affine(self):
        with self.assertRaises(ValueError):
            ScalarImage(tensor=torch.rand(1, 2, 3, 4), affine=np.eye(3))

    def test_nans_tensor(self):
        tensor = np.random.rand(1, 2, 3, 4)
        tensor[0, 0, 0, 0] = np.nan
        with self.assertWarns(UserWarning):
            image = ScalarImage(tensor=tensor, check_nans=True)
        image.set_check_nans(False)

    def test_get_center(self):
        tensor = torch.rand(1, 3, 3, 3)
        image = ScalarImage(tensor=tensor)
        ras = image.get_center()
        lps = image.get_center(lps=True)
        self.assertEqual(ras, (1, 1, 1))
        self.assertEqual(lps, (-1, -1, 1))

    def test_with_list_of_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            ScalarImage(path=['nopath', 'error'])

    def test_with_a_list_of_paths(self):
        shape = (5, 5, 5)
        path1 = self.get_image_path('path1', shape=shape)
        path2 = self.get_image_path('path2', shape=shape)
        image = ScalarImage(path=[path1, path2])
        self.assertEqual(image.shape, (2, 5, 5, 5))
        self.assertEqual(image[STEM], ['path1', 'path2'])

    def test_with_a_list_of_images_with_different_shapes(self):
        path1 = self.get_image_path('path1', shape=(5, 5, 5))
        path2 = self.get_image_path('path2', shape=(7, 5, 5))
        image = ScalarImage(path=[path1, path2])
        with self.assertRaises(RuntimeError):
            image.load()

    def test_with_a_list_of_images_with_different_affines(self):
        path1 = self.get_image_path('path1', spacing=(1, 1, 1))
        path2 = self.get_image_path('path2', spacing=(1, 2, 1))
        image = ScalarImage(path=[path1, path2])
        with self.assertWarns(RuntimeWarning):
            image.load()

    def test_with_a_list_of_2d_paths(self):
        shape = (5, 6)
        path1 = self.get_image_path('path1', shape=shape, suffix='.nii')
        path2 = self.get_image_path('path2', shape=shape, suffix='.img')
        path3 = self.get_image_path('path3', shape=shape, suffix='.hdr')
        image = ScalarImage(path=[path1, path2, path3])
        self.assertEqual(image.shape, (3, 5, 6, 1))
        self.assertEqual(image[STEM], ['path1', 'path2', 'path3'])

    def test_axis_name_2d(self):
        path = self.get_image_path('im2d', shape=(5, 6))
        image = ScalarImage(path)
        height_idx = image.axis_name_to_index('t')
        width_idx = image.axis_name_to_index('l')
        self.assertEqual(image.height, image.shape[height_idx])
        self.assertEqual(image.width, image.shape[width_idx])
