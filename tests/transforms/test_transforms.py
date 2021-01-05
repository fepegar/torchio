import copy
import torch
import numpy as np
import nibabel as nib
import torchio as tio
import SimpleITK as sitk
from ..utils import TorchioTestCase


class TestTransforms(TorchioTestCase):
    """Tests for all transforms."""

    def get_transform(self, channels, is_3d=True, labels=True):
        landmarks_dict = {
            channel: np.linspace(0, 100, 13) for channel in channels
        }
        disp = 1 if is_3d else (1, 1, 0.01)
        elastic = tio.RandomElasticDeformation(max_displacement=disp)
        cp_args = (9, 21, 30) if is_3d else (21, 30, 1)
        flip_axes = axes_downsample = (0, 1, 2) if is_3d else (0, 1)
        swap_patch = (2, 3, 4) if is_3d else (3, 4, 1)
        pad_args = (1, 2, 3, 0, 5, 6) if is_3d else (0, 0, 3, 0, 5, 6)
        crop_args = (3, 2, 8, 0, 1, 4) if is_3d else (0, 0, 8, 0, 1, 4)
        remapping = {1: 2, 2: 1, 3: 20, 4: 25}
        transforms = [
            tio.CropOrPad(cp_args),
            tio.ToCanonical(),
            tio.RandomAnisotropy(downsampling=(1.75, 2), axes=axes_downsample),
            tio.EnsureShapeMultiple(2, method='crop'),
            tio.Resample((1, 1.1, 1.25)),
            tio.RandomFlip(axes=flip_axes, flip_probability=1),
            tio.RandomMotion(),
            tio.RandomGhosting(axes=(0, 1, 2)),
            tio.RandomSpike(),
            tio.RandomNoise(),
            tio.RandomBlur(),
            tio.RandomSwap(patch_size=swap_patch, num_iterations=5),
            tio.Lambda(lambda x: 2 * x, types_to_apply=tio.INTENSITY),
            tio.RandomBiasField(),
            tio.RescaleIntensity((0, 1)),
            tio.ZNormalization(),
            tio.HistogramStandardization(landmarks_dict),
            elastic,
            tio.RandomAffine(),
            tio.OneOf({
                tio.RandomAffine(): 3,
                elastic: 1,
            }),
            tio.RemapLabels(remapping=remapping, masking_method='Left'),
            tio.RemoveLabels([1, 3]),
            tio.SequentialLabels(),
            tio.Pad(pad_args, padding_mode=3),
            tio.Crop(crop_args),
        ]
        if labels:
            transforms.append(tio.RandomLabelsToImage(label_key='label'))
        return tio.Compose(transforms)

    def test_transforms_dict(self):
        transform = tio.RandomNoise(include=('t1', 't2'))
        input_dict = {k: v.data for (k, v) in self.sample_subject.items()}
        transformed = transform(input_dict)
        self.assertIsInstance(transformed, dict)

    def test_transforms_dict_no_keys(self):
        transform = tio.RandomNoise()
        input_dict = {k: v.data for (k, v) in self.sample_subject.items()}
        with self.assertRaises(RuntimeError):
            transform(input_dict)

    def test_transforms_image(self):
        transform = self.get_transform(
            channels=('default_image_name',), labels=False)
        transformed = transform(self.sample_subject.t1)
        self.assertIsInstance(transformed, tio.ScalarImage)

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
        image = tio.data.io.nib_to_sitk(tensor, affine)
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

    def test_transforms_subject_3d(self):
        transform = self.get_transform(channels=('t1', 't2'), is_3d=True)
        transformed = transform(self.sample_subject)
        self.assertIsInstance(transformed, tio.Subject)

    def test_transforms_subject_2d(self):
        transform = self.get_transform(channels=('t1', 't2'), is_3d=False)
        subject = self.make_2d(self.sample_subject)
        transformed = transform(subject)
        self.assertIsInstance(transformed, tio.Subject)

    def test_transforms_subject_4d(self):
        composed = self.get_transform(channels=('t1', 't2'), is_3d=True)
        subject = self.make_multichannel(self.sample_subject)
        subject = self.flip_affine_x(subject)
        transformed = None
        for transform in composed.transforms:
            transformed = transform(subject)
            trsf_channels = len(transformed.t1.data)
            assert trsf_channels > 1, f'Lost channels in {transform.name}'
            exclude = (
                'RandomLabelsToImage',
                'RemapLabels',
                'RemoveLabels',
                'SequentialLabels',
            )
            if transform.name not in exclude:
                self.assertEqual(
                    subject.shape[0],
                    transformed.shape[0],
                    f'Different number of channels after {transform.name}'
                )
                self.assertTensorNotEqual(
                    subject.t1.data[1],
                    transformed.t1.data[1],
                    f'No changes after {transform.name}'
                )
            subject = transformed
        self.assertIsInstance(transformed, tio.Subject)

    def test_transform_noop(self):
        transform = tio.RandomMotion(p=0)
        transformed = transform(self.sample_subject)
        self.assertIs(transformed, self.sample_subject)
        tensor = torch.rand(2, 4, 5, 8).numpy()
        transformed = transform(tensor)
        self.assertIs(transformed, tensor)

    def test_original_unchanged(self):
        subject = copy.deepcopy(self.sample_subject)
        composed = self.get_transform(channels=('t1', 't2'), is_3d=True)
        subject = self.flip_affine_x(subject)
        for transform in composed.transforms:
            original_data = copy.deepcopy(subject.t1.data)
            transform(subject)
            self.assertTensorEqual(
                subject.t1.data,
                original_data,
                f'Changes after {transform.name}'
            )

    def test_transforms_use_include(self):
        original_subject = copy.deepcopy(self.sample_subject)
        transform = tio.RandomNoise(include=['t1'])
        transformed = transform(self.sample_subject)

        self.assertTensorNotEqual(
            original_subject.t1.data,
            transformed.t1.data,
            f'Changes after {transform.name}'
        )

        self.assertTensorEqual(
            original_subject.t2.data,
            transformed.t2.data,
            f'Changes after {transform.name}'
        )

    def test_transforms_use_exclude(self):
        original_subject = copy.deepcopy(self.sample_subject)
        transform = tio.RandomNoise(exclude=['t2'])
        transformed = transform(self.sample_subject)

        self.assertTensorNotEqual(
            original_subject.t1.data,
            transformed.t1.data,
            f'Changes after {transform.name}'
        )

        self.assertTensorEqual(
            original_subject.t2.data,
            transformed.t2.data,
            f'Changes after {transform.name}'
        )

    def test_transforms_use_include_and_exclude(self):
        with self.assertRaises(ValueError):
            tio.RandomNoise(include=['t2'], exclude=['t1'])

    def test_keys_deprecated(self):
        with self.assertWarns(DeprecationWarning):
            tio.RandomNoise(keys=['t2'])

    def test_keep_original(self):
        subject = copy.deepcopy(self.sample_subject)
        old, new = 't1', 't1_original'
        transformed = tio.RandomAffine(keep={old: new})(subject)
        assert old in transformed
        assert new in transformed
        self.assertTensorEqual(
            transformed[new].data,
            subject[old].data,
        )
        self.assertTensorNotEqual(
            transformed[new].data,
            transformed[old].data,
        )


class TestTransform(TorchioTestCase):

    def test_abstract_transform(self):
        with self.assertRaises(TypeError):
            tio.Transform()

    def test_arguments_are_not_dict(self):
        transform = tio.Noise(0, 1, 0)
        assert not transform.arguments_are_dict()

    def test_arguments_are_dict(self):
        transform = tio.Noise({'im': 0}, {'im': 1}, {'im': 0})
        assert transform.arguments_are_dict()

    def test_arguments_are_and_are_not_dict(self):
        transform = tio.Noise(0, {'im': 1}, {'im': 0})
        with self.assertRaises(ValueError):
            transform.arguments_are_dict()
