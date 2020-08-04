import urllib.parse
from torchvision.datasets.utils import download_url
from .. import Subject, ScalarImage, LabelMap, DATA_REPO
from ..utils import get_torchio_cache_dir
from ..data.io import read_matrix


class FPG(Subject):
    """Some images of myself for testing."""
    def __init__(self):
        repo_dir = urllib.parse.urljoin(DATA_REPO, 'fernando/')

        self.filenames = {
            't1': 't1.nii.gz',
            'seg': 't1_seg_gif.nii.gz',
            'rigid': 't1_to_mni.tfm',
            'affine': 't1_to_mni_affine.h5',
        }

        download_root = get_torchio_cache_dir() / 'fpg'
        for filename in self.filenames.values():
            stem = filename.split('.')[0]
            download_url(
                urllib.parse.urljoin(repo_dir, filename),
                download_root,
                filename=filename,
            )

        rigid = read_matrix(download_root / self.filenames['rigid'])
        affine = read_matrix(download_root / self.filenames['affine'])
        subject_dict = {
            't1': ScalarImage(
                download_root / self.filenames['t1'],
                rigid_matrix=rigid,
                affine_matrix=affine,
            ),
            'seg': LabelMap(
                download_root / self.filenames['seg'],
                rigid_matrix=rigid,
                affine_matrix=affine,
            ),
        }
        super().__init__(subject_dict)
