from ... import Subject
from ...utils import get_torchio_cache_dir


class SubjectMNI(Subject):
    """Atlases from the Montreal Neurological Institute (MNI).

    See `the website <http://nist.mni.mcgill.ca/?page_id=714>`_
    for more information.
    """
    url_base = 'http://packages.bic.mni.mcgill.ca/mni-models/'

    @property
    def download_root(self):
        return get_torchio_cache_dir() / self.name
