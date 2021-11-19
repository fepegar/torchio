import copy
import torch
import numpy as np
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
        resize_args = (10, 20, 30) if is_3d else (10, 20, 1)
        flip_axes = axes_downsample = (0, 1, 2) if is_3d else (0, 1)
        swap_patch = (2, 3, 4) if is_3d else (3, 4, 1)
        pad_args = (1, 2, 3, 0, 5, 6) if is_3d else (0, 0, 3, 0, 5, 6)
        crop_args = (3, 2, 8, 0, 1, 4) if is_3d else (0, 0, 8, 0, 1, 4)
        remapping = {1: 2, 2: 1, 3: 20, 4: 25}
        transforms = [
            tio.CropOrPad(cp_args),
            tio.EnsureShapeMultiple(2, method='crop'),
            tio.Resize(resize_args),
            tio.ToCanonical(),
            tio.RandomAnisotropy(downsampling=(1.75, 2), axes=axes_downsample),
            tio.CopyAffine(channels[0]),
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
            tio.RescaleIntensity(out_min_max=(0, 1)),
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
            repr(transform)  # cover __repr__
            transformed = transform(subject)
            trsf_channels = len(transformed.t1.data)
            assert trsf_channels > 1, f'Lost channels in {transform.name}'
            exclude = (
                'RandomLabelsToImage',
                'RemapLabels',
                'RemoveLabels',
                'SequentialLabels',
                'CopyAffine',
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
        with self.assertWarns(UserWarning):
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

    def test_bad_over_max(self):
        transform = tio.RandomNoise()
        with self.assertRaises(ValueError):
            transform._parse_range(2, 'name', max_constraint=1)

    def test_bad_over_max_range(self):
        transform = tio.RandomNoise()
        with self.assertRaises(ValueError):
            transform._parse_range((0, 2), 'name', max_constraint=1)

    def test_bad_type(self):
        transform = tio.RandomNoise()
        with self.assertRaises(ValueError):
            transform._parse_range(2.5, 'name', type_constraint=int)

    def test_no_numbers(self):
        transform = tio.RandomNoise()
        with self.assertRaises(ValueError):
            transform._parse_range('j', 'name')

    def test_apply_transform_missing(self):
        class T(tio.Transform):
            pass
        with self.assertRaises(TypeError):
            T().apply_transform(0)

    def test_non_invertible(self):
        transform = tio.RandomBlur()
        with self.assertRaises(RuntimeError):
            transform.inverse()

    def test_batch_history(self):
        # https://github.com/fepegar/torchio/discussions/743
        subject = self.sample_subject
        transform = tio.Compose([
            tio.RandomAffine(),
            tio.CropOrPad(5),
            tio.OneHot(),
        ])
        dataset = tio.SubjectsDataset([subject], transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=tio.utils.history_collate
        )
        batch = tio.utils.get_first_item(loader)
        transformed: tio.Subject = tio.utils.get_subjects_from_batch(batch)[0]
        inverse = transformed.apply_inverse_transform()
        images1 = subject.get_images(intensity_only=False)
        images2 = inverse.get_images(intensity_only=False)
        for image1, image2 in zip(images1, images2):
            assert image1.shape == image2.shape

    def test_bad_bounds_mask(self):
        transform = tio.ZNormalization(masking_method='test')
        with self.assertRaises(ValueError):
            transform(self.sample_subject)

    def test_bounds_mask(self):
        transform = tio.ZNormalization()
        with self.assertRaises(ValueError):
            transform.get_mask_from_anatomical_label('test', 0)
        tensor = torch.rand((1, 2, 2, 2))

        def get_mask(label):
            mask = transform.get_mask_from_anatomical_label(label, tensor)
            return mask

        left = get_mask('Left')
        assert left[:, 0].sum() == 4 and left[:, 1].sum() == 0
        right = get_mask('Right')
        assert right[:, 1].sum() == 4 and right[:, 0].sum() == 0
        posterior = get_mask('Posterior')
        assert posterior[:, :, 0].sum() == 4 and posterior[:, :, 1].sum() == 0
        anterior = get_mask('Anterior')
        assert anterior[:, :, 1].sum() == 4 and anterior[:, :, 0].sum() == 0
        inferior = get_mask('Inferior')
        assert inferior[..., 0].sum() == 4 and inferior[..., 1].sum() == 0
        superior = get_mask('Superior')
        assert superior[..., 1].sum() == 4 and superior[..., 0].sum() == 0

        mask = transform.get_mask_from_bounds(3 * (0, 1), tensor)
        assert mask[0, 0, 0, 0] == 1
        assert mask.sum() == 1
