import urllib.parse
from torchvision.datasets.utils import download_url
from .. import Subject, Image
from ..utils import get_torchio_cache_dir


SLICER_URL = 'https://github.com/Slicer/SlicerTestingData/releases/download/'
URLS_DICT = {
    'MRHead': ('MRHead.nrrd', 'SHA256/cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93'),
}


class Slicer(Subject):
    """Sample data provided by 3D Slicer.

    See `the website <https://www.slicer.org/wiki/SampleData>`_
    for more information.
    """
    def __init__(self, name='MRHead'):
        filename, url_file = URLS_DICT[name]
        url = urllib.parse.urljoin(SLICER_URL, url_file)
        download_root = get_torchio_cache_dir() / 'slicer'
        stem = filename.split('.')[0]
        download_url(
            url,
            download_root,
            filename=filename,
        )
        super().__init__({
            stem: Image(download_root / filename),
        })
