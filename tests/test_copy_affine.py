"""Tests for CopyAffine transform."""

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


class TestCopyAffine:
    def test_copies_affine(self) -> None:
        t1 = tio.ScalarImage(torch.rand(1, 5, 5, 5))
        t2 = tio.ScalarImage(torch.rand(1, 5, 5, 5))
        t2.affine._matrix[0, 3] = 99.0
        subject = tio.Subject(t1=t1, t2=t2)
        result = tio.CopyAffine(target="t1")(subject)
        torch.testing.assert_close(
            result.t2.affine._matrix,
            result.t1.affine._matrix,
        )

    def test_missing_target_raises(self) -> None:
        subject = _make_subject(with_label=False)
        with pytest.raises(KeyError, match="not_here"):
            tio.CopyAffine(target="not_here")(subject)

    def test_does_not_modify_target(self) -> None:
        subject = _make_subject()
        original = subject.t1.affine._matrix.clone()
        tio.CopyAffine(target="t1")(subject)
        torch.testing.assert_close(subject.t1.affine._matrix, original)

    def test_respects_exclude(self) -> None:
        t1 = tio.ScalarImage(torch.rand(1, 5, 5, 5))
        t2 = tio.ScalarImage(torch.rand(1, 5, 5, 5))
        t2.affine._matrix[0, 3] = 99.0
        original_t2_affine = t2.affine._matrix.clone()
        subject = tio.Subject(t1=t1, t2=t2)
        result = tio.CopyAffine(target="t1", exclude=["t2"])(subject)
        torch.testing.assert_close(result.t2.affine._matrix, original_t2_affine)
