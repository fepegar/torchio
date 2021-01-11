import copy
import tempfile
import pytest
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
    def test_same_space(self):
        # https://github.com/fepegar/torchio/issues/381
        affine1 = np.array([
            [ 4.27109375e-14, -8.71264808e-03,  9.99876633e-01, -3.39850907e+01],
            [-5.54687500e-01, -2.71630469e-12,  8.75148028e-17, 1.62282930e+02],
            [ 2.71575000e-12, -5.54619070e-01, -1.57073092e-02, 2.28515784e+02],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00],
        ])
        affine2 = np.array([
            [ 3.67499773e-08, -8.71257665e-03,  9.99876635e-01, -3.39850922e+01],
            [-5.54687500e-01,  3.67499771e-08,  6.73024385e-08, 1.62282928e+02],
            [-3.73318194e-08, -5.54619071e-01, -1.57071802e-02, 2.28515778e+02],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00],
        ])
        t = torch.rand(1, 2, 3, 4)
        subject = tio.Subject(
            im1=tio.ScalarImage(tensor=t, affine=affine1),
            im2=tio.ScalarImage(tensor=t, affine=affine2),
        )
        subject.check_consistent_space()

    def test_delete_image(self):
        subject = copy.deepcopy(self.sample_subject)
        subject.remove_image('t1')
        with self.assertRaises(KeyError):
            subject['t1']
        with self.assertRaises(AttributeError):
            subject.t1
