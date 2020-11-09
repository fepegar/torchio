import SimpleITK as sitk
from torchio.utils import sitk_to_nib
from torchio.transforms import Pad
from ...utils import TorchioTestCase


class TestPad(TorchioTestCase):
    """Tests for `Pad`."""
    def test_pad(self):
        image = self.sample_subject.t1
        padding = 1, 2, 3, 4, 5, 6
        sitk_image = image.as_sitk()
        low, high = padding[::2], padding[1::2]
        sitk_padded = sitk.ConstantPad(sitk_image, low, high, 0)
        tio_padded = Pad(padding, padding_mode=0)(image)
        sitk_tensor, sitk_affine = sitk_to_nib(sitk_padded)
        tio_tensor, tio_affine = sitk_to_nib(tio_padded.as_sitk())
        self.assertTensorEqual(sitk_tensor, tio_tensor)
        self.assertTensorEqual(sitk_affine, tio_affine)
