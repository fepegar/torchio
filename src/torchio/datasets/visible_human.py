import tempfile
from typing import Tuple

from .. import ScalarImage
from ..data.subject import _RawSubjectCopySubject
from ..download import download_and_extract_archive
from ..utils import get_torchio_cache_dir


class VisibleHuman(_RawSubjectCopySubject):

    URL = 'https://mri.radiology.uiowa.edu/website_documents/visible_human_tar_files/{}{}.tar.gz'  # noqa: E501, FS003
    PARTS: Tuple[str, ...]

    def __init__(self, part: str):
        self.part = self._parse_part(part)
        if not self.cache_part_dir.is_dir():
            tempdir = tempfile.gettempdir()
            filename = f'{self.__class__.__name__}-{self.part}.tar.gz'
            download_and_extract_archive(
                self.url,
                tempdir,
                filename=filename,
                extract_root=self.cache_class_dir,
                remove_finished=True,
            )
        super().__init__({self.part.lower(): ScalarImage(self.cache_part_dir)})

    @property
    def cache_class_dir(self):
        return get_torchio_cache_dir() / self.__class__.__name__

    @property
    def cache_part_dir(self):
        return self.cache_class_dir / self.part

    @property
    def url(self):
        return self.URL.format(self.PREFIX, self.part)

    def _parse_part(self, part: str) -> str:
        part_capital = part.capitalize()
        if part_capital not in self.PARTS:  # type: ignore[assignment]
            message = f'Part "{part}" not in available parts: {self.PARTS}'
            raise ValueError(message)
        return part_capital


class VisibleMale(VisibleHuman):
    """Visible Male CT Datasets.

    Args:
        part: Can be ``'Head'``, ``'Hip'``, ``'Pelvis'`` or ``'Shoulder'``.
    """

    PREFIX = 'VHMCT1mm_'
    PARTS = (
        'Head',
        'Hip',
        'Pelvis',
        'Shoulder',
    )


class VisibleFemale(VisibleHuman):
    """Visible Female CT Datasets.

    Args:
        part: Can be ``'Ankle'``, ``'Head'``, ``'Hip'``, ``'Knee'``,
            ``'Pelvis'`` or ``'Shoulder'``.
    """

    PREFIX = 'VHF-'
    PARTS = VisibleMale.PARTS + (
        'Ankle',
        'Knee',
    )
