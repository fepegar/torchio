import urllib.parse
from torchvision.datasets.utils import download_and_extract_archive
from ...utils import get_torchio_cache_dir
from ... import ScalarImage, LabelMap
from .mni import SubjectMNI


SUPPORTED_YEARS = (
    (4.5, 18.5),
    (4.5, 8.5),
    (7, 11),
    (7.5, 13.5),
    (10, 14),
    (13, 18.5),
)


def format_age(n):
    integer = int(n)
    decimal = int(10 * (n - integer))
    return f'{integer:02d}.{decimal}'


class Pediatric(SubjectMNI):
    """MNI pediatric atlases.

    See `the MNI website <http://nist.mni.mcgill.ca/?p=974>`_ for more information.

    Arguments:
        years: Tuple of 2 ages. Possible values are: ``(4.5, 18.5)``,
            ``(4.5, 8.5)``,
            ``(7, 11)``,
            ``(7.5, 13.5)``,
            ``(10, 14)`` and
            ``(13, 18.5)``.
        symmetric: If ``True``, the left-right symmetric templates will be used.
            If ``False``, the asymmetric (natural) templates will be used.
    """
    def __init__(self, years, symmetric=False):
        self.url_dir = 'http://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1/'
        sym_string = 'sym' if symmetric else 'asym'
        if not isinstance(years, tuple) or years not in SUPPORTED_YEARS:
            message = f'Years must be a tuple in {SUPPORTED_YEARS}'
            raise ValueError(message)
        a, b = years
        file_id = f'{sym_string}_{format_age(a)}-{format_age(b)}'
        self.name = f'nihpd_{file_id}_nifti'
        self.filename = f'{self.name}.zip'
        self.url = urllib.parse.urljoin(self.url_dir, self.filename)
        download_root = get_torchio_cache_dir() / self.name
        if download_root.is_dir():
            print(f'Using cache found in {download_root}')
        else:
            download_and_extract_archive(
                self.url,
                download_root=download_root,
                filename=self.filename,
            )
        super().__init__(
            t1=ScalarImage(download_root / f'nihpd_{file_id}_t1w.nii'),
            t2=ScalarImage(download_root / f'nihpd_{file_id}_t2w.nii'),
            pd=ScalarImage(download_root / f'nihpd_{file_id}_pdw.nii'),
            mask=LabelMap(download_root / f'nihpd_{file_id}_mask.nii'),
        )
