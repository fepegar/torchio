"""Tests for the Reorient transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio
from torchio.data.affine import AffineMatrix


def _make_subject(
    shape: tuple[int, int, int] = (10, 20, 30),
    orientation: str = "RAS",
) -> tio.Subject:
    """Create a subject with a known orientation."""
    import nibabel as nib

    # Build an affine for the desired orientation
    target_ornt = nib.orientations.axcodes2ornt(tuple(orientation))
    # Start from identity and apply the orientation
    base_affine = nib.orientations.inv_ornt_aff(target_ornt, shape)
    affine = AffineMatrix(base_affine)
    data = torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.float32)
    data = data.reshape(1, *shape)
    image = tio.ScalarImage(data, affine=affine)
    return tio.Subject(t1=image)


# ---------------------------------------------------------------------------
# Basic reorientation
# ---------------------------------------------------------------------------


class TestReorientBasic:
    def test_no_op_when_already_target(self) -> None:
        subject = _make_subject(orientation="RAS")
        result = tio.Reorient(orientation="RAS")(subject)
        assert result.t1.affine.orientation == ("R", "A", "S")
        torch.testing.assert_close(result.t1.data, subject.t1.data)

    def test_ras_to_las(self) -> None:
        subject = _make_subject(orientation="RAS")
        result = tio.Reorient(orientation="LAS")(subject)
        assert result.t1.affine.orientation == ("L", "A", "S")

    def test_ras_to_pls(self) -> None:
        subject = _make_subject(orientation="RAS")
        result = tio.Reorient(orientation="PLS")(subject)
        assert result.t1.affine.orientation == ("P", "L", "S")

    def test_default_is_ras(self) -> None:
        subject = _make_subject(orientation="LAS")
        result = tio.Reorient()(subject)
        assert result.t1.affine.orientation == ("R", "A", "S")

    def test_shape_changes_with_permutation(self) -> None:
        subject = _make_subject(shape=(10, 20, 30), orientation="RAS")
        result = tio.Reorient(orientation="ASR")(subject)
        assert result.t1.affine.orientation == ("A", "S", "R")
        # Spatial shape should be a permutation of (10, 20, 30)
        assert sorted(result.t1.spatial_shape) == [10, 20, 30]


# ---------------------------------------------------------------------------
# Round-trip: reorient and back preserves data
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_ras_to_las_and_back(self) -> None:
        subject = _make_subject(orientation="RAS")
        original_data = subject.t1.data.clone()
        to_las = tio.Reorient(orientation="LAS")
        to_ras = tio.Reorient(orientation="RAS")
        result = to_ras(to_las(subject))
        torch.testing.assert_close(result.t1.data, original_data)

    def test_ras_to_spl_and_back(self) -> None:
        subject = _make_subject(orientation="RAS")
        original_data = subject.t1.data.clone()
        result = tio.Reorient(orientation="RAS")(
            tio.Reorient(orientation="SPL")(subject)
        )
        torch.testing.assert_close(result.t1.data, original_data)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_length(self) -> None:
        with pytest.raises(ValueError, match="3-letter"):
            tio.Reorient(orientation="RA")

    def test_invalid_characters(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            tio.Reorient(orientation="XYZ")

    def test_missing_axis(self) -> None:
        with pytest.raises(ValueError, match="each axis"):
            tio.Reorient(orientation="RAA")

    def test_case_insensitive(self) -> None:
        transform = tio.Reorient(orientation="ras")
        assert transform.orientation == "RAS"


# ---------------------------------------------------------------------------
# All images in subject
# ---------------------------------------------------------------------------


class TestAllImages:
    def test_reorients_all_images(self) -> None:
        import nibabel as nib

        ornt = nib.orientations.axcodes2ornt(tuple("RAS"))
        affine = AffineMatrix(nib.orientations.inv_ornt_aff(ornt, (10, 20, 30)))
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 10, 20, 30), affine=affine),
            seg=tio.LabelMap(
                torch.randint(0, 3, (1, 10, 20, 30)),
                affine=affine,
            ),
        )
        result = tio.Reorient(orientation="LAS")(subject)
        assert result.t1.affine.orientation == ("L", "A", "S")
        assert result.seg.affine.orientation == ("L", "A", "S")

    def test_images_with_different_orientations(self) -> None:
        subject = tio.Subject(
            t1=_make_subject(orientation="RAS").t1,
            t2=_make_subject(orientation="LPS").t1,
            seg=_make_subject(orientation="PIL").t1,
        )
        result = tio.Reorient(orientation="SPL")(subject)
        for name in ("t1", "t2", "seg"):
            assert result[name].affine.orientation == ("S", "P", "L")
            TestWorldCoordinates()._assert_bbox_preserved(subject[name], result[name])

    def test_inverse_with_different_orientations(self) -> None:
        subject = tio.Subject(
            t1=_make_subject(orientation="RAS").t1,
            t2=_make_subject(orientation="PIL").t1,
        )
        transformed = tio.Reorient(orientation="LPS")(subject)
        restored = transformed.apply_inverse_transform()
        for name in ("t1", "t2"):
            assert restored[name].affine == subject[name].affine
            torch.testing.assert_close(restored[name].data, subject[name].data)

    def test_inverse_skips_images_added_later(self) -> None:
        subject = _make_subject(orientation="RAS")
        transformed = tio.Reorient(orientation="LPS")(subject)
        added = _make_subject(orientation="PIL").t1
        extended = tio.Subject(t1=transformed.t1, t2=added)
        extended.applied_transforms = transformed.applied_transforms
        restored = extended.apply_inverse_transform()
        assert restored.t1.affine == subject.t1.affine
        assert restored.t2.affine == added.affine
        torch.testing.assert_close(restored.t2.data, added.data)


# ---------------------------------------------------------------------------
# Invertibility
# ---------------------------------------------------------------------------


class TestInvertibility:
    def test_invertible(self) -> None:
        assert tio.Reorient().invertible

    def test_inverse_restores_shape(self) -> None:
        subject = _make_subject(shape=(10, 20, 30), orientation="RAS")
        original_data = subject.t1.data.clone()
        transformed = tio.Reorient(orientation="SPL")(subject)
        restored = transformed.apply_inverse_transform()
        assert restored.t1.spatial_shape == (10, 20, 30)
        torch.testing.assert_close(restored.t1.data, original_data)


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


class TestInputTypes:
    def test_accepts_image(self) -> None:
        import nibabel as nib

        ornt = nib.orientations.axcodes2ornt(tuple("RAS"))
        affine = AffineMatrix(nib.orientations.inv_ornt_aff(ornt, (10, 20, 30)))
        image = tio.ScalarImage(
            torch.rand(1, 10, 20, 30),
            affine=affine,
        )
        result = tio.Reorient(orientation="LAS")(image)
        assert isinstance(result, tio.Image)
        assert result.affine.orientation == ("L", "A", "S")

    def test_accepts_subject(self) -> None:
        subject = _make_subject(orientation="RAS")
        result = tio.Reorient(orientation="LAS")(subject)
        assert isinstance(result, tio.Subject)


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_reorient(self) -> None:
        import nibabel as nib

        from torchio.data.batch import SubjectsBatch

        ornt = nib.orientations.axcodes2ornt(tuple("RAS"))
        affine = AffineMatrix(nib.orientations.inv_ornt_aff(ornt, (10, 20, 30)))
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(
                    torch.rand(1, 10, 20, 30),
                    affine=affine,
                ),
            )
            for _ in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        result = tio.Reorient(orientation="LAS")(batch)
        assert result.t1.data.shape == (3, 1, 10, 20, 30)


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------


class TestProbability:
    def test_p_zero_is_no_op(self) -> None:
        subject = _make_subject(orientation="RAS")
        result = tio.Reorient(orientation="LAS", p=0)(subject)
        assert result.t1.affine.orientation == ("R", "A", "S")


# ---------------------------------------------------------------------------
# World coordinate preservation
# ---------------------------------------------------------------------------


class TestWorldCoordinates:
    """Reorientation must not change the physical position of voxels."""

    @staticmethod
    def _world_bbox(image: tio.Image) -> torch.Tensor:
        s = image.spatial_shape
        corners = torch.tensor(
            [[0, 0, 0, 1], [s[0] - 1, s[1] - 1, s[2] - 1, 1]],
            dtype=torch.float64,
        )
        aff = torch.as_tensor(image.affine.data, dtype=torch.float64)
        return ((aff @ corners.T).T)[:, :3]

    def _assert_bbox_preserved(
        self,
        original: tio.Image,
        reoriented: tio.Image,
    ) -> None:
        orig = self._world_bbox(original).sort(dim=0).values
        new = self._world_bbox(reoriented).sort(dim=0).values
        torch.testing.assert_close(orig, new, atol=1e-5, rtol=0)

    @pytest.mark.parametrize("target", ["PSR", "LPS", "SLA", "AIR", "RAS"])
    def test_bbox_preserved_identity_spacing(self, target: str) -> None:
        subject = _make_subject(shape=(10, 12, 14), orientation="RAS")
        result = tio.Reorient(target)(subject)
        self._assert_bbox_preserved(subject.t1, result.t1)

    def test_bbox_preserved_anisotropic_spacing(self) -> None:
        import numpy as np

        affine = np.diag([2.0, 0.5, 1.5, 1.0])
        affine[:3, 3] = [10, 20, 30]
        data = torch.rand(1, 10, 12, 14)
        img = tio.ScalarImage(data, affine=AffineMatrix(affine))
        subject = tio.Subject(t1=img)
        result = tio.Reorient("PSR")(subject)
        self._assert_bbox_preserved(subject.t1, result.t1)
