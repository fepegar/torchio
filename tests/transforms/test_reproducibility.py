import warnings
import torch
import torchio
import numpy as np
from torchio import Subject, ScalarImage, LabelMap
from torchio.transforms import RandomNoise, compose_from_history, Compose, RandomSpike, OneOf
from ..utils import TorchioTestCase


class TestReproducibility(TorchioTestCase):

    def get_subjects(self):
        subject1 = Subject(img=ScalarImage(tensor=torch.ones(1, 4, 4, 4)))
        subject2 = Subject(img=ScalarImage(tensor=torch.ones(1, 4, 4, 4)))
        return subject1, subject2

    def apply_transforms(self, subject, trsfm_list, seeds_list):
        s = subject
        for trsfm, seed in zip(trsfm_list, seeds_list):
            if seed:
                s = trsfm(s, seed=seed)
            else:
                s = trsfm(s)
        return s

    def random_stuff(self, seed=42):
        transform = RandomNoise(std=(100, 100))
        transformed = transform(self.sample_subject, seed=seed)
        value = transformed.img.data.sum().item()
        seed = transformed.history[0][1]['seed']
        return value, seed

    def test_rng_state(self):
        trsfm = RandomNoise()
        subject1, subject2 = self.get_subjects()
        transformed1 = trsfm(subject1)
        seed1 = transformed1.history[0][1]['seed']
        value1_torch, value1_np = torch.rand(1).item(), np.random.rand()
        transformed2 = trsfm(subject2, seed=seed1)
        value2_torch, value2_np = torch.rand(1).item(), np.random.rand()
        data1, data2 = transformed1.img.data, transformed2.img.data
        self.assertNotEqual(value1_torch, value2_torch)
        self.assertNotEqual(value1_np, value2_np)
        self.assertTensorEqual(data1, data2)

    def test_reproducibility_seed(self):
        trsfm = RandomNoise()
        subject1, subject2 = self.get_subjects()
        transformed1 = trsfm(subject1)
        seed1 = transformed1.history[0][1]['seed']
        transformed2 = trsfm(subject2, seed=seed1)
        data1, data2 = transformed1.img.data, transformed2.img.data
        seed2 = transformed2.history[0][1]['seed']
        self.assertTensorEqual(data1, data2)
        self.assertEqual(seed1, seed2)

    def test_reproducibility_no_seed(self):
        trsfm = RandomNoise()
        subject1, subject2 = self.get_subjects()
        transformed1 = trsfm(subject1)
        transformed2 = trsfm(subject2)
        data1, data2 = transformed1.img.data, transformed2.img.data
        seed1, seed2 = transformed1.history[0][1]['seed'], transformed2.history[0][1]['seed']
        self.assertNotEqual(seed1, seed2)
        self.assertTensorNotEqual(data1, data2)

    def test_reproducibility_from_history(self):
        trsfm = RandomNoise()
        subject1, subject2 = self.get_subjects()
        transformed1 = trsfm(subject1)
        history1 = transformed1.history
        trsfm_hist, seeds_hist = compose_from_history(history=history1)
        transformed2 = self.apply_transforms(subject2, trsfm_list=trsfm_hist, seeds_list=seeds_hist)
        data1, data2 = transformed1.img.data, transformed2.img.data
        self.assertTensorEqual(data1, data2)

    def test_reproducibility_compose(self):
        trsfm = Compose([RandomNoise(p=0.0), RandomSpike(num_spikes=3, p=1.0)])
        subject1, subject2 = self.get_subjects()
        transformed1 = trsfm(subject1)
        history1 = transformed1.history
        trsfm_hist, seeds_hist = compose_from_history(history=history1)
        transformed2 = self.apply_transforms(subject2, trsfm_list=trsfm_hist, seeds_list=seeds_hist)
        data1, data2 = transformed1.img.data, transformed2.img.data
        self.assertTensorEqual(data1, data2)

    def test_reproducibility_oneof(self):
        subject1, subject2 = self.get_subjects()
        trsfm = Compose([OneOf([RandomNoise(p=1.0), RandomSpike(num_spikes=3, p=1.0)]), RandomNoise(p=.5)])
        transformed1 = trsfm(subject1)
        history1 = transformed1.history
        trsfm_hist, seeds_hist = compose_from_history(history=history1)
        transformed2 = self.apply_transforms(subject2, trsfm_list=trsfm_hist, seeds_list=seeds_hist)
        data1, data2 = transformed1.img.data, transformed2.img.data
        self.assertTensorEqual(data1, data2)

    def test_all_random_transforms(self):
        sample = Subject(
            t1=ScalarImage(tensor=torch.rand(1, 20, 20, 20)),
            seg=LabelMap(tensor=torch.rand(1, 20, 20, 20) > 1)
        )

        transforms_names = [
            name
            for name in dir(torchio)
            if name.startswith('Random')
        ]

        # Downsample at the end so that image shape is not modified
        transforms_names.remove('RandomDownsample')
        transforms_names.append('RandomDownsample')

        transforms = []
        for transform_name in transforms_names:
            # Only transform needing an argument for __init__
            if transform_name == 'RandomLabelsToImage':
                transform = getattr(torchio, transform_name)(label_key='seg')
            else:
                transform = getattr(torchio, transform_name)()
            transforms.append(transform)
        composed_transform = torchio.Compose(transforms)
        with warnings.catch_warnings():  # ignore elastic deformation warning
            warnings.simplefilter('ignore', RuntimeWarning)
            transformed = composed_transform(sample)

        new_transforms, seeds = compose_from_history(transformed.history)
        new_transformed = self.apply_transforms(subject=sample, trsfm_list=new_transforms, seeds_list=seeds)
        """
        new_transforms = []
        seeds = []

        for transform_name, params_dict in transformed.history:
            # The Resample transform in the history comes from the DownSampling
            if transform_name in ['Resample', 'Compose']:
                continue
            transform_class = getattr(torchio, transform_name)

            if transform_name == 'RandomLabelsToImage':
                transform = transform_class(label_key='seg')
            else:
                transform = transform_class()
            new_transforms.append(transform)
            seeds.append(params_dict['seed'])

        composed_transform = torchio.Compose(new_transforms)
        with warnings.catch_warnings():  # ignore elastic deformation warning
            warnings.simplefilter('ignore', RuntimeWarning)
            new_transformed = composed_transform(sample, seeds=seeds)
        """

        self.assertTensorEqual(transformed.t1.data, new_transformed.t1.data)
        self.assertTensorEqual(transformed.seg.data, new_transformed.seg.data)
