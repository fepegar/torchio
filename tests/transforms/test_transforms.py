import copy

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
import torch
import torchio as tio

from ..utils import TorchioTestCase


class TestTransforms(TorchioTestCase):
    """Tests for all transforms."""

    def get_transform(self, channels, is_3d=True, labels=True):
        landmarks_dict = {channel: np.linspace(0, 100, 13) for channel in channels}
        disp = 1 if is_3d else (1, 1, 0.01)
        elastic = tio.RandomElasticDeformation(max_displacement=disp)
        affine_elastic = tio.RandomCombinedAffineElasticDeformation(
            elastic_kwargs={'max_displacement': disp}
        )
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
            affine_elastic,
            tio.OneOf(
                {
                    tio.RandomAffine(): 3,
                    elastic: 1,
                }
            ),
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
        assert isinstance(transformed, dict)

    def test_transforms_dict_no_keys(self):
        transform = tio.RandomNoise()
        input_dict = {k: v.data for (k, v) in self.sample_subject.items()}
        with pytest.raises(RuntimeError):
            transform(input_dict)

    def test_transforms_image(self):
        transform = self.get_transform(
            channels=('default_image_name',),
            labels=False,
        )
        transformed = transform(self.sample_subject.t1)
        assert isinstance(transformed, tio.ScalarImage)

    def test_transforms_tensor(self):
        tensor = torch.rand(2, 4, 5, 8)
        transform = self.get_transform(
            channels=('default_image_name',),
            labels=False,
        )
        transformed = transform(tensor)
        assert isinstance(transformed, torch.Tensor)

    def test_transforms_array(self):
        tensor = torch.rand(2, 4, 5, 8).numpy()
        transform = self.get_transform(
            channels=('default_image_name',),
            labels=False,
        )
        transformed = transform(tensor)
        assert isinstance(transformed, np.ndarray)

    def test_transforms_sitk(self):
        tensor = torch.rand(2, 4, 5, 8)
        affine = np.diag((-1, 2, -3, 1))
        image = tio.data.io.nib_to_sitk(tensor, affine)
        transform = self.get_transform(
            channels=('default_image_name',),
            labels=False,
        )
        transformed = transform(image)
        assert isinstance(transformed, sitk.Image)

    def test_transforms_subject_3d(self):
        transform = self.get_transform(channels=('t1', 't2'), is_3d=True)
        transformed = transform(self.sample_subject)
        assert isinstance(transformed, tio.Subject)

    def test_transforms_subject_2d(self):
        transform = self.get_transform(channels=('t1', 't2'), is_3d=False)
        subject = self.make_2d(self.sample_subject)
        transformed = transform(subject)
        assert isinstance(transformed, tio.Subject)

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
                assert (
                    subject.shape[0] == transformed.shape[0]
                ), f'Different number of channels after {transform.name}'
                self.assert_tensor_not_equal(
                    subject.t1.data[1],
                    transformed.t1.data[1],
                    msg=f'No changes after {transform.name}',
                )
            subject = transformed
        assert isinstance(transformed, tio.Subject)

    def test_transform_noop(self):
        transform = tio.RandomMotion(p=0)
        transformed = transform(self.sample_subject)
        assert transformed is self.sample_subject
        tensor = torch.rand(2, 4, 5, 8).numpy()
        transformed = transform(tensor)
        assert transformed is tensor

    def test_original_unchanged(self):
        subject = copy.deepcopy(self.sample_subject)
        composed = self.get_transform(channels=('t1', 't2'), is_3d=True)
        subject = self.flip_affine_x(subject)
        for transform in composed.transforms:
            original_data = copy.deepcopy(subject.t1.data)
            transform(subject)
            self.assert_tensor_equal(
                subject.t1.data,
                original_data,
                msg=f'Changes after {transform.name}',
            )

    def test_transforms_use_include(self):
        original_subject = copy.deepcopy(self.sample_subject)
        transform = tio.RandomNoise(include=['t1'])
        transformed = transform(self.sample_subject)

        self.assert_tensor_not_equal(
            original_subject.t1.data,
            transformed.t1.data,
            msg=f'Changes after {transform.name}',
        )

        self.assert_tensor_equal(
            original_subject.t2.data,
            transformed.t2.data,
            msg=f'Changes after {transform.name}',
        )

    def test_transforms_use_exclude(self):
        original_subject = copy.deepcopy(self.sample_subject)
        transform = tio.RandomNoise(exclude=['t2'])
        transformed = transform(self.sample_subject)

        self.assert_tensor_not_equal(
            original_subject.t1.data,
            transformed.t1.data,
            msg=f'Changes after {transform.name}',
        )

        self.assert_tensor_equal(
            original_subject.t2.data,
            transformed.t2.data,
            msg=f'Changes after {transform.name}',
        )

    def test_transforms_use_include_and_exclude(self):
        with pytest.raises(ValueError):
            tio.RandomNoise(include=['t2'], exclude=['t1'])

    def test_keys_deprecated(self):
        with pytest.warns(DeprecationWarning):
            tio.RandomNoise(keys=['t2'])

    def test_keep_original(self):
        subject = copy.deepcopy(self.sample_subject)
        old, new = 't1', 't1_original'
        transformed = tio.RandomAffine(keep={old: new})(subject)
        assert old in transformed
        assert new in transformed
        self.assert_tensor_equal(
            transformed[new].data,
            subject[old].data,
        )
        self.assert_tensor_not_equal(
            transformed[new].data,
            transformed[old].data,
        )


class TestTransform(TorchioTestCase):
    def test_abstract_transform(self):
        with pytest.raises(TypeError):
            tio.Transform()

    def test_arguments_are_not_dict(self):
        transform = tio.Noise(0, 1, 0)
        assert not transform.arguments_are_dict()

    def test_arguments_are_dict(self):
        transform = tio.Noise({'im': 0}, {'im': 1}, {'im': 0})
        assert transform.arguments_are_dict()

    def test_arguments_are_and_are_not_dict(self):
        transform = tio.Noise(0, {'im': 1}, {'im': 0})
        with pytest.raises(ValueError):
            transform.arguments_are_dict()

    def test_bad_over_max(self):
        transform = tio.RandomNoise()
        with pytest.raises(ValueError):
            transform._parse_range(2, 'name', max_constraint=1)

    def test_bad_over_max_range(self):
        transform = tio.RandomNoise()
        with pytest.raises(ValueError):
            transform._parse_range((0, 2), 'name', max_constraint=1)

    def test_bad_type(self):
        transform = tio.RandomNoise()
        with pytest.raises(ValueError):
            transform._parse_range(2.5, 'name', type_constraint=int)

    def test_no_numbers(self):
        transform = tio.RandomNoise()
        with pytest.raises(ValueError):
            transform._parse_range('j', 'name')

    def test_apply_transform_missing(self):
        class T(tio.Transform):
            pass

        with pytest.raises(TypeError):
            T().apply_transform(0)

    def test_non_invertible(self):
        transform = tio.RandomBlur()
        with pytest.raises(RuntimeError):
            transform.inverse()

    def test_batch_history(self):
        # https://github.com/fepegar/torchio/discussions/743
        subject = self.sample_subject
        transform = tio.Compose(
            [
                tio.RandomAffine(),
                tio.CropOrPad(5),
                tio.OneHot(),
            ]
        )
        dataset = tio.SubjectsDataset([subject], transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=tio.utils.history_collate,
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
        with pytest.raises(ValueError):
            transform(self.sample_subject)

    def test_bounds_mask(self):
        transform = tio.ZNormalization()
        with pytest.raises(ValueError):
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

    def test_label_keys(self):
        # Adapted from the issue in which the feature was requested:
        # https://github.com/fepegar/torchio/issues/866#issue-1222255576
        size = 1, 10, 10, 10
        image = torch.rand(size)
        num_classes = 2  # excluding background
        label = torch.randint(num_classes + 1, size)

        data_dict = {'image': image, 'label': label}

        transform = tio.RandomAffine(
            include=['image', 'label'],
            label_keys=['label'],
        )
        transformed_label = transform(data_dict)['label']

        # If the image is indeed transformed as a label map, nearest neighbor
        # interpolation is used by default and therefore no intermediate values
        # can exist in the output
        num_unique_values = len(torch.unique(transformed_label))
        assert num_unique_values <= num_classes + 1

    def test_nibabel_input(self):
        image = self.sample_subject.t1
        image_nib = nib.Nifti1Image(image.data[0].numpy(), image.affine)
        transformed = tio.RandomAffine()(image_nib)
        transformed.get_fdata()
        transformed.affine

        image = self.subject_4d.t1
        tensor_5d = image.data[np.newaxis].permute(2, 3, 4, 0, 1)
        image_nib = nib.Nifti1Image(tensor_5d.numpy(), image.affine)
        transformed = tio.RandomAffine()(image_nib)
        transformed.get_fdata()
        transformed.affine

    def test_bad_shape(self):
        tensor = torch.rand(1, 2, 3)
        with pytest.raises(ValueError, match='must be a 4D tensor'):
            tio.RandomAffine()(tensor)

    def test_bad_keys_type(self):
        # From https://github.com/fepegar/torchio/issues/923
        with self.assertRaises(ValueError):
            tio.RandomAffine(include='t1')
