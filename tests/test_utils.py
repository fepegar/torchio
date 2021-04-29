import torch
import torchio as tio
from torchio.utils import (
    to_tuple,
    get_stem,
    guess_type,
    apply_transform_to_file,
)
from .utils import TorchioTestCase


class TestUtils(TorchioTestCase):
    """Tests for `utils` module."""

    def test_to_tuple(self):
        assert to_tuple(1) == (1,)
        assert to_tuple((1,)) == (1,)
        assert to_tuple(1, length=3) == (1, 1, 1)
        assert to_tuple((1, 2)) == (1, 2)
        assert to_tuple((1, 2), length=3) == (1, 2)
        assert to_tuple([1, 2], length=3) == (1, 2)

    def test_get_stem(self):
        assert get_stem('/home/image.nii.gz') == 'image'
        assert get_stem('/home/image.nii') == 'image'
        assert get_stem('/home/image.nrrd') == 'image'

    def test_guess_type(self):
        assert guess_type('None') is None
        assert isinstance(guess_type('1'), int)
        assert isinstance(guess_type('1.5'), float)
        assert isinstance(guess_type('(1, 3, 5)'), tuple)
        assert isinstance(guess_type('(1,3,5)'), tuple)
        assert isinstance(guess_type('[1,3,5]'), list)
        assert isinstance(guess_type('test'), str)

    def test_apply_transform_to_file(self):
        transform = tio.RandomFlip()
        apply_transform_to_file(
            self.get_image_path('input'),
            transform,
            self.get_image_path('output'),
            verbose=True,
        )

    def test_subjects_from_batch(self):
        dataset = tio.SubjectsDataset(4 * [self.sample_subject])
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        batch = tio.utils.get_first_item(loader)
        subjects = tio.utils.get_subjects_from_batch(batch)
        assert isinstance(subjects[0], tio.Subject)

    def test_empty_batch(self):
        with self.assertRaises(RuntimeError):
            tio.utils.get_batch_images_and_size({})
