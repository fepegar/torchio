"""
Most of this code is from torchvision.
I will remove all this once verbosity is reduced.
More info: https://github.com/pytorch/vision/issues/2830
"""

import os
import gzip
import urllib
import zipfile
import hashlib
import tarfile
from typing import Optional

from torch.hub import tqdm

from .typing import TypePath


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


# Adapted from torchvision, removing print statements
def download_and_extract_archive(
        url: str,
        download_root: TypePath,
        extract_root: Optional[TypePath] = None,
        filename: Optional[TypePath] = None,
        md5: str = None,
        remove_finished: bool = False,
        ) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)
    download_url(url, download_root, filename, md5)
    archive = os.path.join(download_root, filename)
    extract_archive(archive, extract_root, remove_finished)


def _is_tarxz(filename):
    return filename.endswith('.tar.xz')


def _is_tar(filename):
    return filename.endswith('.tar')


def _is_targz(filename):
    return filename.endswith('.tar.gz')


def _is_tgz(filename):
    return filename.endswith('.tgz')


def _is_gzip(filename):
    return filename.endswith('.gz') and not filename.endswith('.tar.gz')


def _is_zip(filename):
    return filename.endswith('.zip')


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        stem = os.path.splitext(os.path.basename(from_path))[0]
        to_path = os.path.join(to_path, stem)
        with open(to_path, 'wb') as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError(f'Extraction of {from_path} not supported')

    if remove_finished:
        os.remove(from_path)


# Adapted from torchvision, removing print statements
def download_url(
        url: str,
        root: TypePath,
        filename: Optional[TypePath] = None,
        md5: str = None,
        ) -> None:
    """Download a file from a url and place it in root.

    Args:
        url: URL to download file from
        root: Directory to place downloaded file in
        filename: Name to save the file under.
            If ``None``, use the basename of the URL
        md5: MD5 checksum of the download. If None, do not check
    """

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    os.makedirs(root, exist_ok=True)
    # check if file is already present locally
    if not check_integrity(fpath, md5):
        try:
            print('Downloading ' + url + ' to ' + fpath)  # noqa: T001
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, OSError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                message = (
                    'Failed download. Trying https -> http instead.'
                    ' Downloading ' + url + ' to ' + fpath
                )
                print(message)  # noqa: T001
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError('File not found or corrupted.')
