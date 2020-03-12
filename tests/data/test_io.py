import tempfile
import unittest
from pathlib import Path
import SimpleITK as sitk
from ..utils import TorchioTestCase
from torchio.data import io


class TestIO(TorchioTestCase):
    """Tests for `io` module."""
    def setUp(self):
        super().setUp()
        self.write_dicom()

    def write_dicom(self):
        self.dicom_dir = self.dir / 'dicom'
        self.dicom_dir.mkdir()
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
        io.read_image(self.nii_path, itk_first=True)

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
