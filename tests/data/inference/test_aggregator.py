import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestAggregator(TorchioTestCase):
    """Tests for `aggregator` module."""

    def aggregate(self, mode, fixture):
        image_shape = 1, 1, 4, 4
        tensor = torch.ones(image_shape)
        fixture = torch.as_tensor(fixture).reshape(image_shape)
        image_name = 'img'
        subject = tio.Subject({image_name: tio.ScalarImage(tensor=tensor)})
        patch_size = 1, 3, 3
        patch_overlap = 0, 2, 2
        sampler = tio.data.GridSampler(subject, patch_size, patch_overlap)
        aggregator = tio.data.GridAggregator(sampler, overlap_mode=mode)
        loader = tio.SubjectsLoader(sampler, batch_size=3)
        values_dict = {
            (0, 0): 0,
            (0, 1): 2,
            (1, 0): 4,
            (1, 1): 6,
        }
        for batch in loader:
            iterable = zip(batch[tio.LOCATION], batch[image_name][tio.DATA])
            for location, data in iterable:
                coords_2d = tuple(location[1:3].tolist())
                data *= values_dict[coords_2d]
            batch_data = batch[image_name][tio.DATA]
            aggregator.add_batch(batch_data, batch[tio.LOCATION])
        output = aggregator.get_output_tensor()
        self.assert_tensor_equal(output, fixture)

    def test_overlap_crop(self):
        fixture = (
            (0, 0, 2, 2),
            (0, 0, 2, 2),
            (4, 4, 6, 6),
            (4, 4, 6, 6),
        )
        self.aggregate('crop', fixture)

    def test_overlap_average(self):
        fixture = (
            (0, 1, 1, 2),
            (2, 3, 3, 4),
            (2, 3, 3, 4),
            (4, 5, 5, 6),
        )
        self.aggregate('average', fixture)

    def test_overlap_hann(self):
        fixture = (
            (0 / 3, 2 / 3, 4 / 3, 6 / 3),  # noqa: E201, E241
            (4 / 3, 6 / 3, 8 / 3, 10 / 3),  # noqa: E201, E241
            (8 / 3, 10 / 3, 12 / 3, 14 / 3),  # noqa: E201, E241
            (12 / 3, 14 / 3, 16 / 3, 18 / 3),
        )
        self.aggregate('hann', fixture)

    def run_sampler_aggregator(self, overlap_mode='crop'):
        patch_size = 10
        patch_overlap = 2
        grid_sampler = tio.inference.GridSampler(
            self.sample_subject,
            patch_size,
            patch_overlap,
        )
        patch_loader = tio.SubjectsLoader(grid_sampler)
        aggregator = tio.inference.GridAggregator(
            grid_sampler,
            overlap_mode=overlap_mode,
        )
        for batch in patch_loader:
            data = batch['t1'][tio.DATA].long()
            aggregator.add_batch(data, batch[tio.LOCATION])
        return aggregator

    def test_warning_int64(self):
        aggregator = self.run_sampler_aggregator()
        with pytest.warns(RuntimeWarning):
            aggregator.get_output_tensor()

    def run_patch_crop_issue(self, *, padding_mode):
        # https://github.com/TorchIO-project/torchio/issues/813
        pao, pas, ims, bb1, bb2 = 4, 102, 320, 100, 120

        patch_overlap = pao, 0, 0
        patch_size = pas, 1, 1
        img = torch.zeros((1, ims, 1, 1))
        bbox = [bb1, bb2]

        img[:, bbox[0] : bbox[1]] = 1
        image = tio.LabelMap(tensor=img)
        subject = tio.Subject(image=image)
        grid_sampler = tio.inference.GridSampler(
            subject,
            patch_size,
            patch_overlap,
        )
        patch_loader = tio.SubjectsLoader(grid_sampler)
        aggregator = tio.inference.GridAggregator(grid_sampler)
        for patches_batch in patch_loader:
            input_tensor = patches_batch['image'][tio.DATA]
            locations = patches_batch[tio.LOCATION]
            aggregator.add_batch(input_tensor, locations)
        output_tensor = aggregator.get_output_tensor()
        self.assert_tensor_equal(image.tensor, output_tensor)

    def test_patch_crop_issue_no_padding(self):
        self.run_patch_crop_issue(padding_mode=None)

    def test_patch_crop_issue_padding(self):
        self.run_patch_crop_issue(padding_mode='constant')

    def test_bad_aggregator_shape(self):
        # https://github.com/microsoft/InnerEye-DeepLearning/pull/677/checks?check_run_id=5395915817
        tensor = torch.ones(1, 40, 40, 40)
        image_name = 'img'
        subject = tio.Subject({image_name: tio.ScalarImage(tensor=tensor)})
        patch_size = 40
        patch_overlap = 30
        sampler = tio.data.GridSampler(
            subject,
            patch_size,
            patch_overlap,
            padding_mode='edge',
        )
        aggregator = tio.data.GridAggregator(sampler)
        loader = tio.SubjectsLoader(sampler, batch_size=3)
        for batch in loader:
            input_batch = batch[image_name][tio.DATA]
            crop = tio.CropOrPad(12)
            patches = [crop(patch) for patch in input_batch]
            inference_batch = torch.stack(patches)
            with pytest.raises(RuntimeError):
                aggregator.add_batch(inference_batch, batch[tio.LOCATION])
