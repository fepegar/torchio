import urllib.parse
from torchvision.datasets.utils import download_url
from .. import Subject, ScalarImage
from ..utils import get_torchio_cache_dir


SLICER_URL = 'https://github.com/Slicer/SlicerTestingData/releases/download/'
URLS_DICT = {
    'MRHead': (
        ('MRHead.nrrd',),
        ('SHA256/cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93',),
    ),
    'DTIBrain': (
        ('DTI-Brain.nrrd',),
        ('SHA256/5c78d00c86ae8d968caa7a49b870ef8e1c04525b1abc53845751d8bce1f0b91a',),
    ),
    'DTIVolume': (
        (
            'DTIVolume.raw.gz',
            'DTIVolume.nhdr',
        ),
        (
            'SHA256/d785837276758ddd9d21d76a3694e7fd866505a05bc305793517774c117cb38d',
            'SHA256/67564aa42c7e2eec5c3fd68afb5a910e9eab837b61da780933716a3b922e50fe',
        ),
    ),
}


class Slicer(Subject):
    """Sample data provided by `3D Slicer <https://www.slicer.org/>`_.

    See `the Slicer wiki <https://www.slicer.org/wiki/SampleData>`_
    for more information.

    Args:
        name: One of the keys in :py:attr:`torchio.datasets.slicer.URLS_DICT`.
    """
    def __init__(self, name='MRHead'):
        filenames, url_files = URLS_DICT[name]
        for filename, url_file in zip(filenames, url_files):
            filename = filename.replace('-', '_')
            url = urllib.parse.urljoin(SLICER_URL, url_file)
            download_root = get_torchio_cache_dir() / 'slicer'
            stem = filename.split('.')[0]
            download_url(
                url,
                download_root,
                filename=filename,
            )
        super().__init__({
            stem: ScalarImage(download_root / filename),  # will use the last filename
        })
