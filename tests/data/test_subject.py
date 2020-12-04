#!/usr/bin/env python

"""Tests for Subject."""

import tempfile
import torch
import numpy as np
import torchio as tio
from ..utils import TorchioTestCase


class TestSubject(TorchioTestCase):
    """Tests for `Subject`."""
    def test_positional_args(self):
        with self.assertRaises(ValueError):
            with tempfile.NamedTemporaryFile() as f:
                tio.Subject(tio.ScalarImage(f.name))

    def test_input_dict(self):
        with tempfile.NamedTemporaryFile() as f:
            input_dict = {'image': tio.ScalarImage(f.name)}
            tio.Subject(input_dict)
            tio.Subject(**input_dict)

    def test_no_sample(self):
        with tempfile.NamedTemporaryFile() as f:
            input_dict = {'image': tio.ScalarImage(f.name)}
            subject = tio.Subject(input_dict)
            with self.assertRaises(RuntimeError):
                tio.RandomFlip()(subject)

    def test_history(self):
        transformed = tio.RandomFlip()(self.sample_subject)
        self.assertIs(len(transformed.history), 1)

    def test_inconsistent_shape(self):
        subject = tio.Subject(
            a=tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4)),
            b=tio.ScalarImage(tensor=torch.rand(2, 2, 3, 4)),
        )
        subject.spatial_shape
        with self.assertRaises(RuntimeError):
            subject.shape

    def test_inconsistent_spatial_shape(self):
        subject = tio.Subject(
            a=tio.ScalarImage(tensor=torch.rand(1, 3, 3, 4)),
            b=tio.ScalarImage(tensor=torch.rand(2, 2, 3, 4)),
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
        subject = tio.Subject(t1=tio.ScalarImage(self.get_image_path('t1_plot')))
        subject.plot(show=False)

    # flake8: noqa: E203, E241
    def test_different_space(self):
        affine1 = np.array([
            [ -0.69921875,   0.        ,   0.        , 169.11578369],
            [  0.        ,  -0.69921875,   0.        ,  37.26315689],
            [  0.        ,   0.        ,   0.69999993,  15.30004883],
            [  0.        ,   0.        ,   0.        ,   1.        ],
        ])
        affine2 = np.array([
            [ -0.69921881,   0.        ,   0.        , 169.11578369],
            [  0.        ,  -0.69921881,   0.        ,  37.26315689],
            [  0.        ,   0.        ,   0.69999993,  15.30003738],
            [  0.        ,   0.        ,   0.        ,   1.        ],
        ])
        t = torch.rand(1, 2, 3, 4)
        subject = tio.Subject(
            im1=tio.ScalarImage(tensor=t, affine=affine1),
            im2=tio.ScalarImage(tensor=t, affine=affine2),
        )
        with self.assertRaises(RuntimeError):
            subject.check_consistent_space()
