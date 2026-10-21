"""Tests for the Crop transform."""

from __future__ import annotations

import torch

import torchio as tio


class TestCrop:
    def test_crop_uniform(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 20, 20, 20))
        subject = tio.Subject(t1=image)
        result = tio.Crop(cropping=5)(subject)
        assert result.t1.shape == (1, 10, 10, 10)

    def test_crop_per_axis(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 20, 20, 20))
        subject = tio.Subject(t1=image)
        result = tio.Crop(cropping=(2, 4, 6))(subject)
        assert result.t1.shape == (1, 16, 12, 8)

    def test_crop_six_values(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 20, 20, 20))
        subject = tio.Subject(t1=image)
        result = tio.Crop(cropping=(2, 3, 4, 5, 6, 7))(subject)
        assert result.t1.shape == (1, 15, 11, 7)

    def test_crop_all_images(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 20, 20, 20)),
            seg=tio.LabelMap(torch.randint(0, 3, (1, 20, 20, 20))),
        )
        result = tio.Crop(cropping=5)(subject)
        assert result.t1.shape == (1, 10, 10, 10)
        assert result.seg.shape == (1, 10, 10, 10)

    def test_crop_affine_updated(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 20, 20, 20))
        subject = tio.Subject(t1=image)
        original_origin = subject.t1.affine.origin
        result = tio.Crop(cropping=(5, 0, 0, 0, 0, 0))(subject)
        # Origin should shift along first axis
        assert result.t1.affine.origin[0] != original_origin[0]

    def test_crop_lazy(self, tmp_path) -> None:
        """Cropping an unloaded image should not load the full volume."""
        import nibabel as nib
        import numpy as np

        path = tmp_path / "test.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((20, 20, 20)), np.eye(4)),
            path,
        )
        image = tio.ScalarImage(path)
        subject = tio.Subject(t1=image)
        assert not subject.t1.is_loaded
        result = tio.Crop(cropping=5)(subject)
        assert result.t1.shape == (1, 10, 10, 10)

    def test_crop_preserves_laziness_of_original(self, tmp_path) -> None:
        """Cropping should not load the original image's data."""
        import nibabel as nib
        import numpy as np

        path = tmp_path / "test.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((20, 20, 20)), np.eye(4)),
            path,
        )
        image = tio.ScalarImage(path)
        subject = tio.Subject(t1=image)
        assert not subject.t1.is_loaded
        tio.Crop(cropping=5)(subject)
        assert not subject.t1.is_loaded

    def test_crop_history(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 20, 20, 20))
        subject = tio.Subject(t1=image)
        result = tio.Crop(cropping=5)(subject)
        assert len(result.applied_transforms) == 1
        assert result.applied_transforms[0].name == "Crop"

    def test_crop_accepts_image(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 20, 20, 20))
        result = tio.Crop(cropping=5)(image)
        assert isinstance(result, tio.Image)
        assert result.shape == (1, 10, 10, 10)

    def test_crop_accepts_tensor(self) -> None:
        tensor = torch.rand(1, 20, 20, 20)
        result = tio.Crop(cropping=5)(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 10, 10, 10)

    def test_crop_batch(self) -> None:
        from torchio.data.batch import SubjectsBatch

        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 20, 20, 20)),
            )
            for _ in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        result = tio.Crop(cropping=5)(batch)
        assert result.t1.data.shape == (3, 1, 10, 10, 10)

    def test_crop_inverse_on_image(self) -> None:
        """apply_inverse_transform works directly on a cropped Image."""
        image = tio.ScalarImage(torch.rand(1, 200, 200, 200))
        cropped = tio.Crop(cropping=50)(image)
        assert cropped.shape == (1, 100, 100, 100)
        restored = tio.apply_inverse_transform(cropped)
        assert restored.shape == (1, 200, 200, 200)

    def test_crop_entire_axis_from_the_end(self) -> None:
        """Cropping an amount equal to an axis's full size must empty that
        axis, not silently leave it uncropped (`si - i1` is 0 in this case,
        which a former `... or None` substitution mishandled)."""
        image = tio.ScalarImage(torch.rand(1, 20, 20, 20))
        subject = tio.Subject(t1=image)
        result = tio.Crop(cropping=(0, 20, 0, 0, 0, 0))(subject)
        assert result.t1.shape == (1, 0, 20, 20)

    def test_crop_entire_axis_from_the_end_all_axes(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 20, 20, 20))
        subject = tio.Subject(t1=image)
        result = tio.Crop(cropping=(0, 20, 0, 20, 0, 20))(subject)
        assert result.t1.shape == (1, 0, 0, 0)
