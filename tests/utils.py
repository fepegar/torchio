import copy
import shutil
import random
import tempfile
import unittest
from pathlib import Path
from random import shuffle

import torch
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import torchio as tio


class TorchioTestCase(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        self.dir = Path(tempfile.gettempdir()) / '.torchio_tests'
        self.dir.mkdir(exist_ok=True)
        random.seed(42)
        np.random.seed(42)

        registration_matrix = np.array([
            [1, 0, 0, 10],
            [0, 1, 0, 0],
            [0, 0, 1.2, 0],
            [0, 0, 0, 1]
        ])

        subject_a = tio.Subject(
            t1=tio.ScalarImage(self.get_image_path('t1_a')),
        )
        subject_b = tio.Subject(
            t1=tio.ScalarImage(self.get_image_path('t1_b')),
            label=tio.LabelMap(self.get_image_path('label_b', binary=True)),
        )
        subject_c = tio.Subject(
            label=tio.LabelMap(self.get_image_path('label_c', binary=True)),
        )
        subject_d = tio.Subject(
            t1=tio.ScalarImage(
                self.get_image_path('t1_d'),
                pre_affine=registration_matrix,
            ),
            t2=tio.ScalarImage(self.get_image_path('t2_d')),
            label=tio.LabelMap(self.get_image_path('label_d', binary=True)),
        )
        subject_a4 = tio.Subject(
            t1=tio.ScalarImage(self.get_image_path('t1_a'), components=2),
        )
        self.subjects_list = [
            subject_a,
            subject_a4,
            subject_b,
            subject_c,
            subject_d,
        ]
        self.dataset = tio.SubjectsDataset(self.subjects_list)
        self.sample_subject = self.dataset[-1]  # subject_d

    def make_2d(self, subject):
        subject = copy.deepcopy(subject)
        for image in subject.get_images(intensity_only=False):
            image.set_data(image.data[..., :1])
        return subject

    def make_multichannel(self, subject):
        subject = copy.deepcopy(subject)
        for image in subject.get_images(intensity_only=False):
            image.set_data(torch.cat(4 * (image.data,)))
        return subject

    def flip_affine_x(self, subject):
        subject = copy.deepcopy(subject)
        for image in subject.get_images(intensity_only=False):
            image.affine = np.diag((-1, 1, 1, 1)) @ image.affine
        return subject

    def get_inconsistent_shape_subject(self):
        """Return a subject containing images of different shape."""
        subject = tio.Subject(
            t1=tio.ScalarImage(self.get_image_path('t1_inc')),
            t2=tio.ScalarImage(
                self.get_image_path('t2_inc', shape=(10, 20, 31))),
            label=tio.LabelMap(
                self.get_image_path(
                    'label_inc',
                    shape=(8, 17, 25),
                    binary=True,
                ),
            ),
            label2=tio.LabelMap(
                self.get_image_path(
                    'label2_inc',
                    shape=(18, 17, 25),
                    binary=True,
                ),
            ),
        )
        return subject

    def get_reference_image_and_path(self):
        """Return a reference image and its path"""
        path = self.get_image_path(
            'ref',
            shape=(10, 20, 31),
            spacing=(1, 1, 2),
        )
        image = tio.ScalarImage(path)
        return image, path

    def get_subject_with_partial_volume_label_map(self, components=1):
        """Return a subject with a partial-volume label map."""
        return tio.Subject(
            t1=tio.ScalarImage(
                self.get_image_path('t1_d'),
            ),
            label=tio.LabelMap(
                self.get_image_path(
                    'label_d2', binary=False, components=components
                )
            ),
        )

    def get_subject_with_labels(self, labels):
        return tio.Subject(
            label=tio.LabelMap(
                self.get_image_path(
                    'label_multi', labels=labels
                )
            )
        )

    def get_unique_labels(self, label_map):
        labels = torch.unique(label_map.data)
        labels = {i.item() for i in labels if i != 0}
        return labels

    def tearDown(self):
        """Tear down test fixtures, if any."""
        shutil.rmtree(self.dir)

    def get_ixi_tiny(self):
        root_dir = Path(tempfile.gettempdir()) / 'torchio' / 'ixi_tiny'
        return tio.datasets.IXITiny(root_dir, download=True)

    def get_image_path(
            self,
            stem,
            binary=False,
            labels=None,
            shape=(10, 20, 30),
            spacing=(1, 1, 1),
            components=1,
            add_nans=False,
            suffix=None,
            force_binary_foreground=True,
            ):
        shape = (*shape, 1) if len(shape) == 2 else shape
        data = np.random.rand(components, *shape)
        if binary:
            data = (data > 0.5).astype(np.uint8)
            if not data.sum() and force_binary_foreground:
                data[..., 0] = 1
        elif labels is not None:
            data = (data * (len(labels) + 1)).astype(np.uint8)
            new_data = np.zeros_like(data)
            for i, label in enumerate(labels):
                new_data[data == (i + 1)] = label
                if not (new_data == label).sum():
                    new_data[..., i] = label
            data = new_data
        elif self.flip_coin():  # cast some images
            data *= 100
            dtype = np.uint8 if self.flip_coin() else np.uint16
            data = data.astype(dtype)
        if add_nans:
            data[:] = np.nan
        affine = np.diag((*spacing, 1))
        if suffix is None:
            extensions = '.nii.gz', '.nii', '.nrrd', '.img', '.mnc'
            suffix = random.choice(extensions)
        path = self.dir / f'{stem}{suffix}'
        if self.flip_coin():
            path = str(path)
        image = tio.ScalarImage(
            tensor=data,
            affine=affine,
            check_nans=not add_nans,
        )
        image.save(path)
        return path

    def flip_coin(self):
        return np.random.rand() > 0.5

    def get_tests_data_dir(self):
        return Path(__file__).parent / 'image_data'

    def assertTensorNotEqual(self, *args, **kwargs):  # noqa: N802
        message_kwarg = {'msg': args[2]} if len(args) == 3 else {}
        with self.assertRaises(AssertionError, **message_kwarg):
            self.assertTensorEqual(*args, **kwargs)

    @staticmethod
    def assertTensorEqual(*args, **kwargs):  # noqa: N802
        assert_array_equal(*args, **kwargs)

    @staticmethod
    def assertTensorAlmostEqual(*args, **kwargs):  # noqa: N802
        assert_array_almost_equal(*args, **kwargs)

    def get_large_composed_transform(self):
        all_classes = get_all_random_transforms()
        shuffle(all_classes)
        transforms = [t() for t in all_classes]
        # Hack as default patch size for RandomSwap is 15 and sample_subject
        # is (10, 20, 30)
        for tr in transforms:
            if tr.name == 'RandomSwap':
                tr.patch_size = np.array((10, 10, 10))
        return tio.Compose(transforms)


def get_all_random_transforms():
    transforms_names = [
        name
        for name in dir(tio.transforms)
        if name.startswith('Random')
    ]
    classes = [getattr(tio.transforms, name) for name in transforms_names]
    return classes
