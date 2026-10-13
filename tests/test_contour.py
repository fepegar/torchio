"""Tests for Contour transform."""

from __future__ import annotations

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


class TestContour:
    def test_basic_contour(self) -> None:
        subject = _make_subject()
        result = tio.Contour()(subject)
        unique = result.seg.data.unique().tolist()
        assert set(unique) <= {0.0, 1.0}

    def test_solid_block_has_boundary(self) -> None:
        seg = torch.zeros(1, 10, 10, 10, dtype=torch.float32)
        seg[0, 3:7, 3:7, 3:7] = 1
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result = tio.Contour()(subject)
        assert result.seg.data[0, 4, 5, 5] == 0
        assert result.seg.data[0, 3, 5, 5] == 1

    def test_uniform_label_no_contour(self) -> None:
        seg = torch.ones(1, 10, 10, 10, dtype=torch.float32)
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result = tio.Contour()(subject)
        assert result.seg.data[0, 4, 4, 4] == 0

    def test_leaves_scalar_unchanged(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.Contour()(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_respects_exclude(self) -> None:
        seg = torch.ones(1, 10, 10, 10, dtype=torch.float32)
        subject = tio.Subject(a=tio.LabelMap(seg.clone()), b=tio.LabelMap(seg.clone()))
        original_b = subject.b.data.clone()
        result = tio.Contour(exclude=["b"])(subject)
        torch.testing.assert_close(result.b.data, original_b)
