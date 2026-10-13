"""Tests for LabelsToImage transform."""

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


class TestLabelsToImage:
    def test_generates_image(self) -> None:
        subject = _make_subject()
        result = tio.LabelsToImage(label_key="seg")(subject)
        assert "image_from_labels" in result
        assert result.image_from_labels.data.shape[1:] == (10, 10, 10)

    def test_custom_key(self) -> None:
        subject = _make_subject()
        result = tio.LabelsToImage(label_key="seg", image_key="synth")(subject)
        assert "synth" in result

    def test_auto_detect_label(self) -> None:
        subject = _make_subject()
        result = tio.LabelsToImage()(subject)
        assert "image_from_labels" in result

    def test_ignore_background(self) -> None:
        subject = _make_subject()
        result = tio.LabelsToImage(
            label_key="seg",
            ignore_background=True,
        )(subject)
        bg_mask = subject.seg.data == 0
        bg_values = result.image_from_labels.data[0, bg_mask[0]]
        assert bg_values.abs().max() < 1e-5

    def test_no_label_raises(self) -> None:
        subject = _make_subject(with_label=False)
        with pytest.raises(KeyError, match="No LabelMap"):
            tio.LabelsToImage()(subject)

    def test_missing_key_raises(self) -> None:
        subject = _make_subject()
        with pytest.raises(KeyError, match="nope"):
            tio.LabelsToImage(label_key="nope")(subject)

    def test_respects_exclude_for_autodetect(self) -> None:
        subject = _make_subject()
        with pytest.raises(KeyError, match="No LabelMap"):
            tio.LabelsToImage(exclude=["seg"])(subject)


class TestLabelsToImagePerInstance:
    def _batch(self, batch_size: int = 5) -> tio.SubjectsBatch:
        seg = torch.zeros(1, 10, 10, 10, dtype=torch.float32)
        seg[0, 2:5, 2:5, 2:5] = 1
        seg[0, 6:9, 6:9, 6:9] = 2
        subjects = [
            tio.Subject(seg=tio.LabelMap(seg.clone())) for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    def test_per_instance_means_differ_across_batch(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        transform = tio.LabelsToImage(label_key="seg", default_mean=(0.2, 0.9))
        result = transform(batch)
        params = result.applied_transforms[-1].params
        assert "_batched_keys" in params
        assert len(params["means"]) == batch.batch_size
        means_for_label_1 = [m[1] for m in params["means"]]
        assert len(set(means_for_label_1)) > 1
        assert result.image_from_labels.data.shape[0] == batch.batch_size

    def test_per_instance_false_shares_params(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        transform = tio.LabelsToImage(
            label_key="seg",
            default_mean=(0.2, 0.9),
            per_instance=False,
        )
        result = transform(batch)
        params = result.applied_transforms[-1].params
        assert isinstance(params["means"], dict)

    def test_single_subject_keeps_scalar_params(self) -> None:
        subject = _make_subject()
        result = tio.LabelsToImage(label_key="seg", default_mean=(0.2, 0.9))(subject)
        assert isinstance(result.applied_transforms[-1].params["means"], dict)


class TestLabelsToImagePerElementVectorized:
    def test_each_element_uses_its_own_label_stats(self) -> None:
        # The vectorized per-element generation must give each batch element
        # its own per-label mean (no cross-element contamination), even though
        # the random draw order differs from the old per-element loop.
        size = 16
        label = torch.zeros(1, size, size, size)
        label[0, : size // 2] = 1
        label[0, size // 2 :] = 2
        batch = tio.SubjectsBatch.from_subjects(
            [tio.Subject(seg=tio.LabelMap(label.clone())) for _ in range(3)]
        )
        transform = tio.LabelsToImage(
            label_key="seg",
            image_key="img",
            default_mean=(0.0, 100.0),
            default_std=(0.0, 0.05),
        )
        torch.manual_seed(1)
        result = transform(batch)
        params = result.applied_transforms[-1].params
        assert "_batched_keys" in params
        image = result.img.data
        for index in range(batch.batch_size):
            region_one = image[index, 0, : size // 2]
            region_two = image[index, 0, size // 2 :]
            assert region_one.mean().item() == pytest.approx(
                params["means"][index][1], abs=0.5
            )
            assert region_two.mean().item() == pytest.approx(
                params["means"][index][2], abs=0.5
            )
        # Independent per-element sampling: means vary across the batch.
        label_one_means = {round(params["means"][i][1], 3) for i in range(3)}
        assert len(label_one_means) > 1
