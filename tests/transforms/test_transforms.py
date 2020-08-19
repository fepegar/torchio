import copy
import torch
import torchio
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from ..utils import TorchioTestCase


class TestTransforms(TorchioTestCase):
    """Tests for all transforms."""

    def get_transform(self, channels, is_3d=True, labels=True):
        landmarks_dict = {
            channel: np.linspace(0, 100, 13) for channel in channels
        }
        disp = 1 if is_3d else (1, 1, 0.01)
        elastic = torchio.RandomElasticDeformation(max_displacement=disp)
        cp_args = (9, 21, 30) if is_3d else (21, 30, 1)
        flip_axes = axes_downsample = (0, 1, 2) if is_3d else (0, 1)
        swap_patch = (2, 3, 4) if is_3d else (3, 4, 1)
        pad_args = (1, 2, 3, 0, 5, 6) if is_3d else (0, 0, 3, 0, 5, 6)
        crop_args = (3, 2, 8, 0, 1, 4) if is_3d else (0, 0, 8, 0, 1, 4)
        transforms = [
            torchio.CropOrPad(cp_args),
            torchio.ToCanonical(),
            torchio.RandomDownsample(axes=axes_downsample),
            torchio.Resample((1, 1.1, 1.25)),
            torchio.RandomFlip(axes=flip_axes, flip_probability=1),
            torchio.RandomMotion(),
            torchio.RandomGhosting(axes=(0, 1, 2)),
            torchio.RandomSpike(),
            torchio.RandomNoise(),
            torchio.RandomBlur(),
            torchio.RandomSwap(patch_size=swap_patch, num_iterations=5),
            torchio.Lambda(lambda x: 2 * x, types_to_apply=torchio.INTENSITY),
            torchio.RandomBiasField(),
            torchio.RescaleIntensity((0, 1)),
            torchio.ZNormalization(),
            torchio.HistogramStandardization(landmarks_dict),
            elastic,
            torchio.RandomAffine(),
            torchio.OneOf({
                torchio.RandomAffine(): 3,
                elastic: 1,
            }),
            torchio.Pad(pad_args, padding_mode=3),
            torchio.Crop(crop_args),
        ]
        if labels:
            transforms.append(torchio.RandomLabelsToImage(label_key='label'))
        return torchio.Compose(transforms)

    def test_transforms_dict(self):
        transform = torchio.RandomNoise(keys=('t1', 't2'))
        input_dict = {k: v.data for (k, v) in self.sample.items()}
        transformed = transform(input_dict)
        self.assertIsInstance(transformed, dict)

    def test_transforms_dict_no_keys(self):
        transform = torchio.RandomNoise()
        input_dict = {k: v.data for (k, v) in self.sample.items()}
        with self.assertRaises(RuntimeError):
            transform(input_dict)

    def test_transforms_image(self):
        transform = self.get_transform(
            channels=('default_image_name',), labels=False)
        transformed = transform(self.sample.t1)
        self.assertIsInstance(transformed, torchio.ScalarImage)

    def test_transforms_tensor(self):
        tensor = torch.rand(2, 4, 5, 8)
        transform = self.get_transform(
            channels=('default_image_name',), labels=False)
        transformed = transform(tensor)
        self.assertIsInstance(transformed, torch.Tensor)

    def test_transforms_array(self):
        tensor = torch.rand(2, 4, 5, 8).numpy()
        transform = self.get_transform(
            channels=('default_image_name',), labels=False)
        transformed = transform(tensor)
        self.assertIsInstance(transformed, np.ndarray)

    def test_transforms_sitk(self):
        tensor = torch.rand(2, 4, 5, 8)
        affine = np.diag((-1, 2, -3, 1))
        image = torchio.utils.nib_to_sitk(tensor, affine)
        transform = self.get_transform(
            channels=('default_image_name',), labels=False)
        transformed = transform(image)
        self.assertIsInstance(transformed, sitk.Image)

    def test_transforms_nib(self):
        data = torch.rand(1, 4, 5, 8).numpy()
        affine = np.diag((1, -2, 3, 1))
        image = nib.Nifti1Image(data, affine)
        transform = self.get_transform(
            channels=('default_image_name',), labels=False)
        transformed = transform(image)
        self.assertIsInstance(transformed, nib.Nifti1Image)

    def test_transforms_sample_3d(self):
        transform = self.get_transform(channels=('t1', 't2'), is_3d=True)
        transformed = transform(self.sample)
        self.assertIsInstance(transformed, torchio.Subject)

    def test_transforms_sample_2d(self):
        transform = self.get_transform(channels=('t1', 't2'), is_3d=False)
        sample = self.make_2d(self.sample)
        transformed = transform(sample)
        self.assertIsInstance(transformed, torchio.Subject)

    def test_transforms_sample_4d(self):
        composed = self.get_transform(channels=('t1', 't2'), is_3d=True)
        sample = self.make_multichannel(self.sample)
        sample = self.flip_affine_x(sample)
        for transform in composed.transform.transforms:
            transformed = transform(sample)
            trsf_channels = len(transformed.t1.data)
            assert trsf_channels > 1, f'Lost channels in {transform.name}'
            if transform.name != 'RandomLabelsToImage':
                self.assertEqual(
                    sample.shape[0],
                    transformed.shape[0],
                    f'Different number of channels after {transform.name}'
                )
                self.assertTensorNotEqual(
                    sample.t1.data[1],
                    transformed.t1.data[1],
                    f'No changes after {transform.name}'
                )
            sample = transformed
        self.assertIsInstance(transformed, torchio.Subject)

    def test_transform_noop(self):
        transform = torchio.RandomMotion(p=0)
        transformed = transform(self.sample)
        self.assertIs(transformed, self.sample)
        tensor = torch.rand(2, 4, 5, 8).numpy()
        transformed = transform(tensor)
        self.assertIs(transformed, tensor)

    def test_original_unchanged(self):
        sample = copy.deepcopy(self.sample)
        composed = self.get_transform(channels=('t1', 't2'), is_3d=True)
        sample = self.flip_affine_x(sample)
        for transform in composed.transform.transforms:
            original_data = copy.deepcopy(sample.t1.data)
            transform(sample)
            self.assertTensorEqual(
                sample.t1.data,
                original_data,
                f'Changes after {transform.name}'
            )


class TestTransform(TorchioTestCase):
    def test_abstract_transform(self):
        with self.assertRaises(TypeError):
            torchio.Transform()
