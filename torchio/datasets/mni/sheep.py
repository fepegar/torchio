import shutil
import urllib.parse

from ...data import ScalarImage
from ...utils import get_torchio_cache_dir, compress
from ...download import download_and_extract_archive
from .mni import SubjectMNI


class Sheep(SubjectMNI):

    def __init__(self):
        self.name = 'NIFTI_ovine_05mm'
        self.url_dir = urllib.parse.urljoin(self.url_base, 'sheep/')
        self.filename = f'{self.name}.zip'
        self.url = urllib.parse.urljoin(self.url_dir, self.filename)
        download_root = get_torchio_cache_dir() / self.name
        t1_nii_path = download_root / 'ovine_model_05.nii'
        t1_niigz_path = download_root / 'ovine_model_05.nii.gz'
        if not download_root.is_dir():
            download_and_extract_archive(
                self.url,
                download_root=download_root,
                filename=self.filename,
            )
            shutil.rmtree(download_root / 'masks')
            for path in download_root.iterdir():
                if path == t1_nii_path:
                    compress(t1_nii_path, t1_niigz_path)
                path.unlink()
        super().__init__(
            t1=ScalarImage(t1_niigz_path)
        )
