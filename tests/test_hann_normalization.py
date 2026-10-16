"""Hann blending must preserve values, including at volume borders."""

import pytest
import torch

import torchio as tio


@pytest.mark.parametrize("overlap", [0, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hann_reconstructs_identity(overlap, dtype) -> None:
    data = torch.linspace(-1, 2, 2 * 9 * 10 * 11, dtype=dtype).reshape(2, 9, 10, 11)
    subject = tio.Subject(t1=tio.ScalarImage(data))
    sampler = tio.GridSampler(subject, patch_size=6, patch_overlap=overlap)
    aggregator = tio.PatchAggregator(subject.spatial_shape, overlap_mode="hann")
    for patch in sampler:
        aggregator.add_batch(patch.t1.data[None], [patch.patch_location])
    torch.testing.assert_close(aggregator.get_output(), data)


@pytest.mark.parametrize("mode", ["average", "hann"])
def test_partial_output_can_be_read_before_adding_more_patches(mode) -> None:
    aggregator = tio.PatchAggregator((4, 4, 8), overlap_mode=mode)
    first = tio.PatchLocation(index=(0, 0, 0), size=(4, 4, 4))
    second = tio.PatchLocation(index=(0, 0, 4), size=(4, 4, 4))
    data = torch.full((1, 2, 4, 4, 4), 3.0)
    aggregator.add_batch({"prediction": data}, [first])
    partial = aggregator.get_output("prediction")
    torch.testing.assert_close(partial[..., :4], data[0])
    assert torch.count_nonzero(partial[..., 4:]) == 0
    aggregator.add_batch({"prediction": data}, [second])
    torch.testing.assert_close(
        aggregator.get_output("prediction"), torch.full((2, 4, 4, 8), 3.0)
    )


def test_hann_normalizes_downsampled_outputs() -> None:
    aggregator = tio.PatchAggregator(
        (8, 8, 8), overlap_mode="hann", output_shape=(4, 4, 4)
    )
    location = tio.PatchLocation(index=(0, 0, 0), size=(8, 8, 8))
    data = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
    aggregator.add_batch(data, [location])
    torch.testing.assert_close(aggregator.get_output(), data[0])


def test_hann_blends_overlapping_predictions() -> None:
    aggregator = tio.PatchAggregator((1, 1, 4), overlap_mode="hann")
    locations = [
        tio.PatchLocation(index=(0, 0, 0), size=(1, 1, 3)),
        tio.PatchLocation(index=(0, 0, 1), size=(1, 1, 3)),
    ]
    patches = torch.tensor([1.0, 3.0]).reshape(2, 1, 1, 1, 1).expand(2, 1, 1, 1, 3)
    aggregator.add_batch(patches, locations)
    # The length-three Hann weights are [1/2, 1, 1/2].
    expected = torch.tensor([1.0, 5 / 3, 7 / 3, 3.0]).reshape(1, 1, 1, 4)
    torch.testing.assert_close(aggregator.get_output(), expected)
