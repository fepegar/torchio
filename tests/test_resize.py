"""Tests for Resize transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio


def _make_subject(with_label: bool = True) -> tio.Subject:
    data = torch.rand(1, 10, 10, 10) * 100
    kwargs: dict = {"t1": tio.ScalarImage(data)}
    if with_label:
        seg = torch.zeros(1, 10, 10, 10, dtype=torch.float32)
        seg[0, 2:5, 2:5, 2:5] = 1
        seg[0, 6:9, 6:9, 6:9] = 2
        kwargs["seg"] = tio.LabelMap(seg)
    return tio.Subject(**kwargs)


class TestResize:
    def test_resize_to_target(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Resize(5)(subject)
        assert result.t1.data.shape[1:] == (5, 5, 5)

    def test_resize_anisotropic(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Resize((8, 6, 4))(subject)
        assert result.t1.data.shape[1:] == (8, 6, 4)

    def test_resize_preserves_dtype(self) -> None:
        subject = _make_subject()
        result = tio.Resize(5)(subject)
        assert result.t1.data.dtype == subject.t1.data.dtype

    def test_resize_labels_nearest(self) -> None:
        subject = _make_subject()
        result = tio.Resize(5)(subject)
        unique = result.seg.data.unique().tolist()
        for v in unique:
            assert v == int(v)

    def test_resize_with_labels(self) -> None:
        subject = _make_subject()
        result = tio.Resize(5)(subject)
        assert result.seg.data.shape[1:] == (5, 5, 5)

    def test_resize_data_matches_affine(self) -> None:
        # Intensities equal the world coordinate along the first axis, so
        # after resizing each voxel must hold the coordinate of its center
        ramp = torch.arange(10, dtype=torch.float64).reshape(1, 10, 1, 1)
        affine = torch.diag(torch.tensor([2.0, 1.0, 1.0, 1.0], dtype=torch.float64))
        affine[0, 3] = 3
        image = tio.ScalarImage(2 * ramp.expand(1, 10, 2, 2) + 3, affine=affine)
        result = tio.Resize((4, 2, 2))(tio.Subject(t1=image))
        indices = torch.zeros(4, 3, dtype=torch.float64)
        indices[:, 0] = torch.arange(4)
        world = result.t1.affine.apply(indices)[:, 0]
        torch.testing.assert_close(result.t1.data[0, :, 0, 0], world)
        # The field of view is preserved
        assert result.t1.affine.spacing[0] == pytest.approx(5)
        assert result.t1.affine.origin[0] == pytest.approx(4.5)

    def test_resize_labels_nearest_voxel(self) -> None:
        labels = torch.arange(10, dtype=torch.uint8).reshape(1, 10, 1, 1)
        subject = tio.Subject(seg=tio.LabelMap(labels.expand(1, 10, 2, 2)))
        result = tio.Resize((6, 2, 2))(subject)
        expected = torch.tensor([0, 2, 4, 5, 7, 9], dtype=torch.uint8)
        torch.testing.assert_close(result.seg.data[0, :, 0, 0], expected)
