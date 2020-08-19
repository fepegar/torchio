import tempfile
import unittest
from pathlib import Path
import torch
import pytest
import numpy as np
from numpy.testing import assert_array_equal
import nibabel as nib
import SimpleITK as sitk
from ..utils import TorchioTestCase
from torchio.data import io, ScalarImage


class TestIO(TorchioTestCase):
    """Tests for `io` module."""
    def setUp(self):
        super().setUp()
        self.write_dicom()
        string = (
            '1.5 0.18088 -0.124887 0.65072 '
            '-0.20025 0.965639 -0.165653 -11.6452 '
            '0.0906326 0.18661 0.978245 11.4002 '
            '0 0 0 1 '
        )
        tensor = torch.from_numpy(np.fromstring(string, sep=' ').reshape(4, 4))
        self.matrix = tensor

    def write_dicom(self):
        self.dicom_dir = self.dir / 'dicom'
        self.dicom_dir.mkdir(exist_ok=True)
        self.dicom_path = self.dicom_dir / 'dicom.dcm'
        self.nii_path = self.get_image_path('read_image')
        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(self.dicom_path))
        image = sitk.ReadImage(str(self.nii_path))
        image = sitk.Cast(image, sitk.sitkUInt16)
        image = image[0]  # dicom reader supports 2D only
        writer.Execute(image)

    def test_read_image(self):
        # I need to find something readable by nib but not sitk
        io.read_image(self.nii_path)

    def test_save_rgb(self):
        im = ScalarImage(tensor=torch.rand(1, 4, 5, 1))
        with self.assertWarns(UserWarning):
            im.save(self.dir / 'test.jpg')

    def test_read_dicom_file(self):
        io.read_image(self.dicom_path)

    def test_read_dicom_dir(self):
        io.read_image(self.dicom_dir)

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
    assert_array_equal(
        tensor.squeeze(), loaded_tensor.squeeze(),
        f'Save lib: {save_lib}; load lib: {load_lib}; dims: {dims}'
    )
    assert_array_equal(affine, loaded_affine)
