"""Tests for dataset download transport handling."""

from ssl import SSLCertVerificationError
from unittest.mock import Mock
from urllib.error import URLError

import pytest

from torchio import download


def test_https_failure_does_not_retry_over_http(tmp_path, monkeypatch):
    url = "https://example.invalid/image.nii.gz"
    error = URLError(SSLCertVerificationError("certificate verify failed"))
    retrieve = Mock(side_effect=error)
    monkeypatch.setattr(download.request, "urlretrieve", retrieve)

    with pytest.raises(URLError) as caught:
        download.download_url(url, tmp_path)

    assert caught.value is error
    assert retrieve.call_count == 1
    assert retrieve.call_args.args[0] == url
