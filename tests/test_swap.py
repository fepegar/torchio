"""Tests for Swap transform."""

from __future__ import annotations

import warnings

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


class TestSwap:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Swap(patch_size=3, num_iterations=10)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_preserves_shape(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Swap(patch_size=3, num_iterations=5)(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_warns_with_labels(self) -> None:
        subject = _make_subject()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tio.Swap(patch_size=3, num_iterations=1)(subject)
            assert any("LabelMap" in str(warning.message) for warning in w)

    def test_no_warning_when_label_excluded(self) -> None:
        subject = _make_subject()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tio.Swap(patch_size=3, num_iterations=1, exclude=["seg"])(subject)
            assert not any("LabelMap" in str(warning.message) for warning in w)

    def test_patch_too_large_raises(self) -> None:
        subject = _make_subject(with_label=False)
        with pytest.raises(ValueError, match="cannot be larger"):
            tio.Swap(patch_size=100, num_iterations=1)(subject)

    def test_single_iteration(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Swap(patch_size=3, num_iterations=1)(subject)
        assert result.t1.data.shape == subject.t1.data.shape


class TestSwapPerInstance:
    def _batch(self, batch_size: int = 6) -> tio.SubjectsBatch:
        data = torch.rand(1, 16, 16, 16)
        subjects = [
            tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    def test_per_instance_differs_across_batch(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        result = tio.Swap(patch_size=4, num_iterations=20)(batch)
        params = result.applied_transforms[-1].params
        assert "_batched_keys" in params
        assert len(params["locations"]) == batch.batch_size
        assert not torch.allclose(result.t1.data[0], result.t1.data[1])

    def test_per_instance_false_is_shared(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        result = tio.Swap(patch_size=4, num_iterations=20, per_instance=False)(batch)
        torch.testing.assert_close(result.t1.data[0], result.t1.data[1])

    def test_single_subject_keeps_scalar_params(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 16, 16, 16)))
        result = tio.Swap(patch_size=4, num_iterations=20)(subject)
        assert "_batched_keys" not in result.applied_transforms[-1].params


class TestSwapGatedOut:
    def test_gated_out_elements_are_exact_no_ops(self) -> None:
        torch.manual_seed(0)
        data = torch.rand(1, 16, 16, 16)
        batch = tio.SubjectsBatch.from_subjects(
            [tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(32)]
        )
        original = batch.t1.data.clone()
        result = tio.Swap(patch_size=4, num_iterations=20, p=0.5)(batch)
        unchanged = [
            torch.equal(result.t1.data[i], original[i]) for i in range(batch.batch_size)
        ]
        assert any(unchanged)
        assert not all(unchanged)
