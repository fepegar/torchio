#!/usr/bin/env python

"""Tests for Subject."""

import tempfile
import torch
from torchio import Subject, ScalarImage, RandomFlip
from ..utils import TorchioTestCase


class TestSubject(TorchioTestCase):
    """Tests for `Subject`."""
    def test_positional_args(self):
        with self.assertRaises(ValueError):
            with tempfile.NamedTemporaryFile() as f:
                Subject(ScalarImage(f.name))

    def test_input_dict(self):
        with tempfile.NamedTemporaryFile() as f:
            input_dict = {'image': ScalarImage(f.name)}
            Subject(input_dict)
            Subject(**input_dict)

    def test_no_sample(self):
        with tempfile.NamedTemporaryFile() as f:
            input_dict = {'image': ScalarImage(f.name)}
            subject = Subject(input_dict)
            with self.assertRaises(RuntimeError):
                RandomFlip()(subject)

    def test_history(self):
        transformed = RandomFlip()(self.sample_subject)
        self.assertIs(len(transformed.history), 1)

    def test_inconsistent_shape(self):
        subject = Subject(
            a=ScalarImage(tensor=torch.rand(1, 2, 3, 4)),
            b=ScalarImage(tensor=torch.rand(2, 2, 3, 4)),
        )
        subject.spatial_shape
        with self.assertRaises(RuntimeError):
            subject.shape

    def test_inconsistent_spatial_shape(self):
        subject = Subject(
            a=ScalarImage(tensor=torch.rand(1, 3, 3, 4)),
            b=ScalarImage(tensor=torch.rand(2, 2, 3, 4)),
        )
        with self.assertRaises(RuntimeError):
            subject.spatial_shape

    def test_plot(self):
        self.sample_subject.plot(
            show=False,
            output_path=self.dir / 'figure.png',
            cmap_dict=dict(
                t2='viridis',
                label={0: 'yellow', 1: 'blue'},
            ),
        )

    def test_plot_one_image(self):
        subject = Subject(t1=ScalarImage(self.get_image_path('t1_plot')))
        subject.plot(show=False)
