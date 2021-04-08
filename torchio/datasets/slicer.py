import urllib.parse
from ..download import download_url
from .. import Subject, ScalarImage
from ..utils import get_torchio_cache_dir


SLICER_URL = 'https://github.com/Slicer/SlicerTestingData/releases/download/'
URLS_DICT = {
    'MRHead': (
        ('MRHead.nrrd',),
        ('SHA256/cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93',),  # noqa: E501
    ),
    'DTIBrain': (
        ('DTI-Brain.nrrd',),
        ('SHA256/5c78d00c86ae8d968caa7a49b870ef8e1c04525b1abc53845751d8bce1f0b91a',),  # noqa: E501
    ),
    'DTIVolume': (
        (
            'DTIVolume.raw.gz',
            'DTIVolume.nhdr',
        ),
        (
            'SHA256/d785837276758ddd9d21d76a3694e7fd866505a05bc305793517774c117cb38d',  # noqa: E501
            'SHA256/67564aa42c7e2eec5c3fd68afb5a910e9eab837b61da780933716a3b922e50fe',  # noqa: E501
        ),
    ),
    'CTChest': (
        ('CT-chest.nrrd',),
        ('SHA256/4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e',),  # noqa: E501
    ),
    'CTACardio': (
        ('CTA-cardio.nrrd',),
        ('SHA256/3b0d4eb1a7d8ebb0c5a89cc0504640f76a030b4e869e33ff34c564c3d3b88ad2',),  # noqa: E501
    ),
}


class Slicer(Subject):
    """Sample data provided by `3D Slicer <https://www.slicer.org/>`_.

    See `the Slicer wiki <https://www.slicer.org/wiki/SampleData>`_
    for more information.

    For information about licensing and permissions, check the `Sample Data
    module <https://github.com/Slicer/Slicer/blob/31c89f230919a953e56f6722718281ce6da49e06/Modules/Scripted/SampleData/SampleData.py#L75-L81>`_.

    Args:
        name: One of the keys in :attr:`torchio.datasets.slicer.URLS_DICT`.
    """  # noqa: E501
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
            stem: ScalarImage(download_root / filename),  # use last filename
        })
