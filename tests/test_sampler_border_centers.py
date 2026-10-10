"""Every center whose entire patch fits in the image is eligible."""

import itertools

import pytest
import torch

import torchio as tio


@pytest.mark.parametrize("sampler_name", ["weighted", "label"])
@pytest.mark.parametrize("patch_size", [(4, 4, 4), (3, 3, 3), (4, 3, 2), (1, 4, 1)])
@pytest.mark.parametrize("whole_volume", [False, True])
def test_sample_at_last_valid_center(sampler_name, patch_size, whole_volume) -> None:
    shape = patch_size if whole_volume else tuple(s + 2 for s in patch_size)
    data = torch.arange(torch.tensor(shape).prod()).reshape(1, *shape).float()
    label = torch.zeros(1, *shape)
    corner = tuple(s - p for s, p in zip(shape, patch_size, strict=True))
    center = tuple(c + p // 2 for c, p in zip(corner, patch_size, strict=True))
    label[(0, *center)] = 1
    subject = tio.Subject(t1=tio.ScalarImage(data), seg=tio.LabelMap(label))
    if sampler_name == "weighted":
        sampler = tio.WeightedSampler(subject, patch_size, probability_map="seg")
    else:
        sampler = tio.LabelSampler(subject, patch_size, label_name="seg")

    patch = next(sampler(subject, num_patches=1))
    assert patch.patch_location.index == corner
    assert patch.t1.spatial_shape == patch_size
    slices = tuple(slice(c, c + p) for c, p in zip(corner, patch_size, strict=True))
    torch.testing.assert_close(patch.t1.data, data[(slice(None), *slices)])


@pytest.mark.parametrize("patch_size", [(2, 2, 2), (3, 3, 3), (4, 3, 2), (1, 4, 1)])
def test_border_mask_matches_valid_patch_corners(patch_size) -> None:
    shape = (5, 6, 7)
    weights = torch.ones(1, *shape)
    subject = tio.Subject(weights=tio.ScalarImage(weights))
    sampler = tio.WeightedSampler(subject, patch_size, probability_map="weights")
    # Enumerate all in-bounds patches independently of border masking.
    expected = torch.zeros(shape)
    corners = (range(s - p + 1) for s, p in zip(shape, patch_size, strict=True))
    for corner in itertools.product(*corners):
        center = tuple(c + p // 2 for c, p in zip(corner, patch_size, strict=True))
        expected[center] = 1
    torch.testing.assert_close(sampler._build_probability_map(), expected)
    torch.testing.assert_close(subject.weights.data, weights)
