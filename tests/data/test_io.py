import tempfile
from pathlib import Path

import torch
import pytest
import numpy as np
import SimpleITK as sitk

from ..utils import TorchioTestCase
from torchio.data import io, ScalarImage
from torchio.data.io import ensure_4d, nib_to_sitk, sitk_to_nib


class TestIO(TorchioTestCase):
    """Tests for `io` module."""
    def setUp(self):
        super().setUp()
        self.nii_path = self.get_image_path('read_image')
        self.dicom_dir = self.get_tests_data_dir() / 'dicom'
        self.dicom_path = self.dicom_dir / 'IMG0001.dcm'
        string = (
            '1.5 0.18088 -0.124887 0.65072 '
            '-0.20025 0.965639 -0.165653 -11.6452 '
            '0.0906326 0.18661 0.978245 11.4002 '
            '0 0 0 1 '
        )
        tensor = torch.as_tensor(np.fromstring(string, sep=' ').reshape(4, 4))
        self.matrix = tensor

    def test_read_image(self):
        # I need to find something readable by nib but not sitk
        io.read_image(self.nii_path)

    def test_save_rgb(self):
        im = ScalarImage(tensor=torch.rand(1, 4, 5, 1))
        with self.assertWarns(RuntimeWarning):
            im.save(self.dir / 'test.jpg')

    def test_read_dicom_file(self):
        tensor, _ = io.read_image(self.dicom_path)
        self.assertEqual(tuple(tensor.shape), (1, 88, 128, 1))

    def test_read_dicom_dir(self):
        tensor, _ = io.read_image(self.dicom_dir)
        self.assertEqual(tuple(tensor.shape), (1, 88, 128, 17))

    def test_dicom_dir_missing(self):
        with self.assertRaises(FileNotFoundError):
            io._read_dicom('missing')

    def test_dicom_dir_no_files(self):
        empty = self.dir / 'empty'
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            io._read_dicom(empty)

    def write_read_matrix(self, suffix):
        out_path = self.dir / f'matrix{suffix}'
        io.write_matrix(self.matrix, out_path)
        matrix = io.read_matrix(out_path)
        assert torch.allclose(matrix, self.matrix)

    def test_matrix_itk(self):
        self.write_read_matrix('.tfm')
        self.write_read_matrix('.h5')

    def test_matrix_txt(self):
        self.write_read_matrix('.txt')

    def test_ensure_4d_5d(self):
        tensor = torch.rand(3, 4, 5, 1, 2)
        assert ensure_4d(tensor).shape == (2, 3, 4, 5)

    def test_ensure_4d_5d_t_gt_1(self):
        tensor = torch.rand(3, 4, 5, 2, 2)
        with self.assertRaises(ValueError):
            ensure_4d(tensor)

    def test_ensure_4d_2d(self):
        tensor = torch.rand(4, 5)
        assert ensure_4d(tensor).shape == (1, 4, 5, 1)

    def test_ensure_4d_2d_3dims_rgb_first(self):
        tensor = torch.rand(3, 4, 5)
        assert ensure_4d(tensor).shape == (3, 4, 5, 1)

    def test_ensure_4d_2d_3dims_rgb_last(self):
        tensor = torch.rand(4, 5, 3)
        assert ensure_4d(tensor).shape == (3, 4, 5, 1)

    def test_ensure_4d_3d(self):
        tensor = torch.rand(4, 5, 6)
        assert ensure_4d(tensor).shape == (1, 4, 5, 6)

    def test_ensure_4d_2_spatial_dims(self):
        tensor = torch.rand(4, 5, 6)
        assert ensure_4d(tensor, num_spatial_dims=2).shape == (4, 5, 6, 1)

    def test_ensure_4d_3_spatial_dims(self):
        tensor = torch.rand(4, 5, 6)
        assert ensure_4d(tensor, num_spatial_dims=3).shape == (1, 4, 5, 6)

    def test_ensure_4d_nd_not_supported(self):
        tensor = torch.rand(1, 2, 3, 4, 5)
        with self.assertRaises(ValueError):
            ensure_4d(tensor)

    def test_sitk_to_nib(self):
        data = np.random.rand(10, 12)
        image = sitk.GetImageFromArray(data)
        tensor, _ = sitk_to_nib(image)
        self.assertAlmostEqual(data.sum(), tensor.sum())


# This doesn't work as a method of the class
libs = 'sitk', 'nibabel'
parameters = []
for save_lib in libs:
    for load_lib in libs:
        for dims in 2, 3, 4:
            parameters.append((save_lib, load_lib, dims))


@pytest.mark.parametrize(('save_lib', 'load_lib', 'dims'), parameters)
def test_write_nd_with_a_read_it_with_b(save_lib, load_lib, dims):
    shape = [1, 4, 5, 6]
    if dims == 2:
        shape[-1] = 1
    elif dims == 4:
        shape[0] = 2
    tensor = torch.randn(*shape)
    affine = np.eye(4)
    tempdir = Path(tempfile.gettempdir()) / '.torchio_tests'
    tempdir.mkdir(exist_ok=True)
    path = tempdir / 'test_io.nii'
    save_function = getattr(io, f'_write_{save_lib}')
    load_function = getattr(io, f'_read_{save_lib}')
    save_function(tensor, affine, path)
    loaded_tensor, loaded_affine = load_function(path)
    TorchioTestCase.assertTensorEqual(
        tensor.squeeze(), loaded_tensor.squeeze(),
        f'Save lib: {save_lib}; load lib: {load_lib}; dims: {dims}'
    )
    TorchioTestCase.assertTensorEqual(affine, loaded_affine)


class TestNibabelToSimpleITK(TorchioTestCase):

    def setUp(self):
        super().setUp()
        self.affine = np.eye(4)

    def test_wrong_num_dims(self):
        with self.assertRaises(ValueError):
            nib_to_sitk(np.random.rand(10, 10), self.affine)

    def test_2d_single(self):
        data = np.random.rand(1, 10, 12, 1)
        image = nib_to_sitk(data, self.affine)
        assert image.GetDimension() == 2
        assert image.GetSize() == (10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 1

    def test_2d_multi(self):
        data = np.random.rand(5, 10, 12, 1)
        image = nib_to_sitk(data, self.affine)
        assert image.GetDimension() == 2
        assert image.GetSize() == (10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 5

    def test_2d_3d_single(self):
        data = np.random.rand(1, 10, 12, 1)
        image = nib_to_sitk(data, self.affine, force_3d=True)
        assert image.GetDimension() == 3
        assert image.GetSize() == (10, 12, 1)
        assert image.GetNumberOfComponentsPerPixel() == 1

    def test_2d_3d_multi(self):
        data = np.random.rand(5, 10, 12, 1)
        image = nib_to_sitk(data, self.affine, force_3d=True)
        assert image.GetDimension() == 3
        assert image.GetSize() == (10, 12, 1)
        assert image.GetNumberOfComponentsPerPixel() == 5

    def test_3d_single(self):
        data = np.random.rand(1, 8, 10, 12)
        image = nib_to_sitk(data, self.affine)
        assert image.GetDimension() == 3
        assert image.GetSize() == (8, 10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 1

    def test_3d_multi(self):
        data = np.random.rand(5, 8, 10, 12)
        image = nib_to_sitk(data, self.affine)
        assert image.GetDimension() == 3
        assert image.GetSize() == (8, 10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 5
