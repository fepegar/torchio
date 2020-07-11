import warnings
import torch
import torchio
from torchio import Subject, Image, INTENSITY
from torchio.transforms import RandomNoise
from ..utils import TorchioTestCase


class TestReproducibility(TorchioTestCase):

    def setUp(self):
        super().setUp()
        self.subject = Subject(img=Image(tensor=torch.ones(4, 4, 4)))

    def random_stuff(self, seed=None):
        transform = RandomNoise(std=(100, 100))#, seed=seed)
        transformed = transform(self.subject, seed=seed)
        value = transformed.img.data.sum().item()
        _, seed = transformed.get_applied_transforms()[0]
        return value, seed

    def test_reproducibility_no_seed(self):
        a, seed_a = self.random_stuff()
        b, seed_b = self.random_stuff()
        self.assertNotEqual(a, b)
        c, seed_c = self.random_stuff(seed_a)
        self.assertEqual(c, a)
        self.assertEqual(seed_c, seed_a)

    def test_reproducibility_seed(self):
        torch.manual_seed(42)
        a, seed_a = self.random_stuff()
        b, seed_b = self.random_stuff()
        self.assertNotEqual(a, b)
        c, seed_c = self.random_stuff(seed_a)
        self.assertEqual(c, a)
        self.assertEqual(seed_c, seed_a)

        torch.manual_seed(42)
        a2, seed_a2 = self.random_stuff()
        self.assertEqual(a2, a)
        self.assertEqual(seed_a2, seed_a)
        b2, seed_b2 = self.random_stuff()
        self.assertNotEqual(a2, b2)
        self.assertEqual(b2, b)
        self.assertEqual(seed_b2, seed_b)
        c2, seed_c2 = self.random_stuff(seed_a2)
        self.assertEqual(c2, a2)
        self.assertEqual(seed_c2, seed_a2)
        self.assertEqual(c2, c)
        self.assertEqual(seed_c2, seed_c)

    # def test_all_random_transforms(self):
    #     sample = Subject(
    #         t1=Image(tensor=torch.rand(20, 20, 20)),
    #         seg=Image(tensor=torch.rand(20, 20, 20) > 1, type=INTENSITY)
    #     )

    #     transforms_names = [
    #         name
    #         for name in dir(torchio)
    #         if name.startswith('Random')
    #     ]

    #     # Downsample at the end so that the image shape is not modified
    #     transforms_names.remove('RandomDownsample')
    #     transforms_names.append('RandomDownsample')

    #     transforms = []
    #     for transform_name in transforms_names:
    #         transform = getattr(torchio, transform_name)()
    #         transforms.append(transform)
    #     composed_transform = torchio.Compose(transforms)
    #     with warnings.catch_warnings():  # ignore elastic deformation warning
    #         warnings.simplefilter('ignore', UserWarning)
    #         transformed = composed_transform(sample)

    #     new_transforms = []
    #     for transform_name, params_dict in transformed.history:
    #         transform_class = getattr(torchio, transform_name)
    #         transform = transform_class(seed=params_dict['seed'])
    #         new_transforms.append(transform)
    #     composed_transform = torchio.Compose(transforms)
    #     with warnings.catch_warnings():  # ignore elastic deformation warning
    #         warnings.simplefilter('ignore', UserWarning)
    #         new_transformed = composed_transform(sample)

    #     self.assertTensorEqual(transformed.t1.data, new_transformed.t1.data)
    #     self.assertTensorEqual(transformed.seg.data, new_transformed.seg.data)
