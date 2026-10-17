"""Tests for Standardize (z-score normalization)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchio as tio


def _make_subject(with_label: bool = False) -> tio.Subject:
    torch.manual_seed(42)
    data = torch.randn(1, 10, 10, 10) * 50 + 100  # mean ~100, std ~50
    kwargs: dict = {"t1": tio.ScalarImage(data)}
    if with_label:
        mask = torch.zeros(1, 10, 10, 10)
        mask[0, 2:8, 2:8, 2:8] = 1
        kwargs["brain"] = tio.LabelMap(mask)
    return tio.Subject(**kwargs)


class TestBasic:
    def test_output_has_zero_mean_unit_std(self) -> None:
        subject = _make_subject()
        result = tio.Standardize()(subject)
        data = result.t1.data
        assert abs(data.mean().item()) < 0.01
        assert abs(data.std().item() - 1.0) < 0.01

    def test_leaves_label_maps_unchanged(self) -> None:
        subject = _make_subject(with_label=True)
        original = subject.brain.data.clone()
        result = tio.Standardize()(subject)
        torch.testing.assert_close(result.brain.data, original)


class TestMasking:
    def test_masking_with_label_key(self) -> None:
        subject = _make_subject(with_label=True)
        result = tio.Standardize(masking_method="brain")(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_masking_with_callable(self) -> None:
        subject = _make_subject()
        result = tio.Standardize(masking_method=lambda x: x > 100)(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_masking_key_not_found(self) -> None:
        subject = _make_subject()
        with pytest.raises(KeyError, match="nonexistent"):
            tio.Standardize(masking_method="nonexistent")(subject)

    def test_masking_not_labelmap(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
            t2=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        with pytest.raises(TypeError, match="LabelMap"):
            tio.Standardize(masking_method="t2")(subject)


class TestEdgeCases:
    def test_zero_std_raises(self) -> None:
        data = torch.ones(1, 4, 4, 4) * 42.0
        subject = tio.Subject(t1=tio.ScalarImage(data))
        with pytest.raises(RuntimeError, match="zero"):
            tio.Standardize()(subject)

    def test_empty_mask_warns(self) -> None:
        subject = _make_subject()
        with pytest.warns(RuntimeWarning, match="empty"):
            tio.Standardize(
                masking_method=lambda x: torch.zeros_like(x, dtype=torch.bool),
            )(subject)


class TestInverse:
    def test_inverse_restores_values(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        transformed = tio.Standardize()(subject)
        restored = transformed.apply_inverse_transform()
        np.testing.assert_allclose(
            restored.t1.data.numpy(),
            original.numpy(),
            atol=1e-4,
        )


class TestExports:
    def test_available_at_top_level(self) -> None:
        assert hasattr(tio, "Standardize")

    def test_znormalization_alias(self) -> None:
        assert tio.ZNormalization is tio.Standardize


class TestBatch:
    def test_stats_are_computed_per_sample(self) -> None:
        torch.manual_seed(0)
        quiet = tio.Subject(t1=tio.ScalarImage(torch.randn(1, 4, 4, 4) + 10))
        loud = tio.Subject(t1=tio.ScalarImage(torch.randn(1, 4, 4, 4) * 50 + 1000))
        batch = tio.SubjectsBatch.from_subjects([quiet, loud])

        data = tio.Standardize()(batch).images["t1"].data

        # Every sample must be standardized with its own mean and std, not
        # with the statistics of the first sample in the batch.
        for index in range(len(data)):
            assert abs(data[index].mean().item()) < 0.01
            assert abs(data[index].std().item() - 1.0) < 0.01

    def test_batch_matches_subject_by_subject(self) -> None:
        torch.manual_seed(0)
        subjects = [
            tio.Subject(t1=tio.ScalarImage(torch.randn(1, 4, 4, 4) * scale + shift))
            for scale, shift in ((1.0, 10.0), (50.0, 1000.0), (0.5, -5.0))
        ]
        batched = tio.Standardize()(tio.SubjectsBatch.from_subjects(subjects))

        for index, subject in enumerate(subjects):
            alone = tio.Standardize()(subject)
            torch.testing.assert_close(
                batched.images["t1"].data[index],
                alone.t1.data,
                atol=1e-5,
                rtol=0,
            )
