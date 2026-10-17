"""Tests for KeepLargestComponent transform."""

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


class TestKeepLargestComponent:
    def test_keeps_largest_binary(self) -> None:
        seg = torch.zeros(1, 20, 20, 20, dtype=torch.float32)
        seg[0, 1:3, 1:3, 1:3] = 1
        seg[0, 10:18, 10:18, 10:18] = 1
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result = tio.KeepLargestComponent()(subject)
        assert result.seg.data[0, 2, 2, 2] == 0
        assert result.seg.data[0, 14, 14, 14] == 1

    def test_multi_label(self) -> None:
        seg = torch.zeros(1, 20, 20, 20, dtype=torch.float32)
        seg[0, 0:2, 0:2, 0:2] = 1
        seg[0, 10:13, 10:13, 10:13] = 1
        seg[0, 5:8, 5:8, 5:8] = 2
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result = tio.KeepLargestComponent()(subject)
        assert result.seg.data[0, 1, 1, 1] == 0
        assert result.seg.data[0, 11, 11, 11] == 1
        assert result.seg.data[0, 6, 6, 6] == 2

    def test_specific_labels(self) -> None:
        seg = torch.zeros(1, 20, 20, 20, dtype=torch.float32)
        seg[0, 0:2, 0:2, 0:2] = 1
        seg[0, 10:13, 10:13, 10:13] = 1
        seg[0, 5:7, 5:7, 5:7] = 2
        seg[0, 15:19, 15:19, 15:19] = 2
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result = tio.KeepLargestComponent(labels=[1])(subject)
        assert result.seg.data[0, 1, 1, 1] == 0
        assert result.seg.data[0, 6, 6, 6] == 2

    def test_multichannel_raises(self) -> None:
        seg = torch.zeros(2, 10, 10, 10, dtype=torch.float32)
        subject = tio.Subject(seg=tio.LabelMap(seg))
        with pytest.raises(RuntimeError, match="single-channel"):
            tio.KeepLargestComponent()(subject)

    def test_leaves_scalar_unchanged(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.KeepLargestComponent()(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_respects_exclude(self) -> None:
        seg = torch.zeros(1, 20, 20, 20, dtype=torch.float32)
        seg[0, 1:3, 1:3, 1:3] = 1
        seg[0, 10:18, 10:18, 10:18] = 1
        subject = tio.Subject(a=tio.LabelMap(seg.clone()), b=tio.LabelMap(seg.clone()))
        original_b = subject.b.data.clone()
        result = tio.KeepLargestComponent(exclude=["b"])(subject)
        torch.testing.assert_close(result.b.data, original_b)

    def test_face_connectivity(self) -> None:
        seg = torch.zeros(1, 10, 10, 10, dtype=torch.float32)
        seg[0, 0:3, 0:3, 0:3] = 1
        seg[0, 3:7, 3:7, 3:7] = 1
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result_26 = tio.KeepLargestComponent(fully_connected=True)(subject)
        assert result_26.seg.data[0, 1, 1, 1] == 1
        assert result_26.seg.data[0, 5, 5, 5] == 1
