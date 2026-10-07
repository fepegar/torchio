"""Image selection for transforms that apply to every image type."""

from __future__ import annotations

import pytest
import torch

import torchio as tio


def _make_subject() -> tio.Subject:
    data = torch.arange(48, dtype=torch.float32).reshape(1, 4, 3, 4)
    return tio.Subject(
        t1=tio.ScalarImage(data.clone()),
        t2=tio.ScalarImage(data.clone()),
    )


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda **kwargs: tio.Lambda(lambda x: -x, **kwargs), id="lambda"),
        pytest.param(tio.Transpose, id="transpose"),
        pytest.param(lambda **kwargs: tio.Resize((2, 3, 2), **kwargs), id="resize"),
        pytest.param(
            lambda **kwargs: tio.Anisotropy(axes=(0,), downsampling=2, **kwargs),
            id="anisotropy",
        ),
    ],
)
@pytest.mark.parametrize(
    "selection",
    [
        pytest.param({"include": ["t1"]}, id="include"),
        pytest.param({"exclude": ["t2"]}, id="exclude"),
    ],
)
def test_unselected_images_are_untouched(factory, selection) -> None:
    subject = _make_subject()
    original = subject.t2.data.clone()
    transformed = factory(**selection)(subject)
    assert not torch.equal(transformed.t1.data, original)
    torch.testing.assert_close(transformed.t2.data, original)
