"""Download utilities for built-in datasets."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib import request

from platformdirs import user_cache_dir
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import Progress
from rich.progress import TransferSpeedColumn

from .types import TypePath


def get_torchio_cache_dir() -> Path:
    """Return the default cache directory for TorchIO data."""
    return Path(user_cache_dir("torchio"))


def calculate_md5(fpath: TypePath, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_integrity(fpath: TypePath, md5: str | None = None) -> bool:
    """Check whether a file exists and optionally matches a checksum."""
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return md5 == calculate_md5(fpath)


def download_url(
    url: str,
    root: TypePath,
    *,
    filename: str | None = None,
    md5: str | None = None,
) -> Path:
    """Download a file from a URL and place it in *root*.

    Args:
        url: URL to download file from.
        root: Directory to place downloaded file in.
        filename: Name to save the file under. If `None`, use the
            basename of the URL.
        md5: MD5 checksum of the download. If `None`, skip check.

    Returns:
        Path to the downloaded file.

    Download errors propagate without retrying HTTPS URLs over HTTP.
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    os.makedirs(root, exist_ok=True)
    if not check_integrity(fpath, md5):
        progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        )

        def _try_download(download_url: str) -> None:
            with progress:
                task = progress.add_task(
                    f"Downloading [cyan]{filename}",
                    total=None,
                )

                def _reporthook(
                    count: int,
                    block_size: int,
                    total_size: int,
                ) -> None:
                    if total_size > 0:
                        progress.update(task, total=total_size)
                    progress.update(task, completed=count * block_size)

                request.urlretrieve(download_url, fpath, reporthook=_reporthook)

        _try_download(url)
        if not check_integrity(fpath, md5):
            msg = f"File not found or corrupted: {fpath}"
            raise RuntimeError(msg)
    return Path(fpath)


def extract_archive(
    from_path: TypePath,
    to_path: TypePath | None = None,
    *,
    remove_finished: bool = False,
) -> None:
    """Extract an archive file.

    Supports `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.xz`,
    and `.gz` (single-file gzip).
    """
    from_path = str(from_path)
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith(".tar"):
        with tarfile.open(from_path, "r") as tar:
            tar.extractall(path=to_path)
    elif from_path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(from_path, "r:gz") as tar:
            tar.extractall(path=to_path)
    elif from_path.endswith(".tar.xz"):
        with tarfile.open(from_path, "r:xz") as tar:
            tar.extractall(path=to_path)
    elif from_path.endswith(".gz"):
        stem = os.path.splitext(os.path.basename(from_path))[0]
        out = os.path.join(str(to_path), stem)
        with open(out, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif from_path.endswith(".zip"):
        with zipfile.ZipFile(from_path, "r") as z:
            z.extractall(to_path)
    else:
        msg = f"Extraction of {from_path} not supported"
        raise ValueError(msg)

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(
    url: str,
    download_root: TypePath,
    *,
    extract_root: TypePath | None = None,
    filename: str | None = None,
    md5: str | None = None,
    remove_finished: bool = False,
) -> None:
    """Download an archive and extract it.

    Args:
        url: URL to download.
        download_root: Directory to place the downloaded archive.
        extract_root: Directory to extract to. Defaults to
            *download_root*.
        filename: Archive filename. Defaults to the URL basename.
        md5: MD5 checksum of the archive.
        remove_finished: Delete the archive after extraction.
    """
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)
    download_url(url, download_root, filename=filename, md5=md5)
    archive = os.path.join(os.path.expanduser(download_root), filename)
    extract_archive(archive, extract_root, remove_finished=remove_finished)


def compress(
    input_path: TypePath,
    output_path: TypePath | None = None,
) -> Path:
    """Compress a NIfTI file with gzip.

    Args:
        input_path: Path to the `.nii` file.
        output_path: Path for the compressed file. Defaults to
            replacing the suffix with `.nii.gz`.

    Returns:
        Path to the compressed file.
    """
    if output_path is None:
        output_path = Path(input_path).with_suffix(".nii.gz")
    with open(input_path, "rb") as f_in, gzip.open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return Path(output_path)
