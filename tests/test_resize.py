"""Tests for Resize transform."""

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

    def test_respects_exclude(self) -> None:
        subject = _make_subject()
        result = tio.Resize(5, exclude=["seg"])(subject)
        assert result.t1.data.shape[1:] == (5, 5, 5)
        assert result.seg.data.shape[1:] == (10, 10, 10)
