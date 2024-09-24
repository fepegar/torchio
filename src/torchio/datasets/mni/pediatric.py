import urllib.parse

from ...data import LabelMap
from ...data import ScalarImage
from ...download import download_and_extract_archive
from ...utils import compress
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

    See `the MNI website <https://nist.mni.mcgill.ca/pediatric-atlases-4-5-18-5y/>`_
    for more information.

    .. image:: https://nist.mni.mcgill.ca/wp-content/uploads/2016/04/nihpd_asym_all_sm.jpg
        :alt: Pediatric MNI template

    Arguments:
        years: Tuple of 2 ages. Possible values are: ``(4.5, 18.5)``,
            ``(4.5, 8.5)``,
            ``(7, 11)``,
            ``(7.5, 13.5)``,
            ``(10, 14)`` and
            ``(13, 18.5)``.
        symmetric: If ``True``, the left-right symmetric templates will be
            used. Else, the asymmetric (natural) templates will be used.
    """

    def __init__(self, years, symmetric=False):
        self.url_dir = 'http://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1/'
        sym_string = 'sym' if symmetric else 'asym'
        if not isinstance(years, tuple) or years not in SUPPORTED_YEARS:
            message = f'Years must be a tuple in {SUPPORTED_YEARS}'
            raise ValueError(message)
        a, b = years
        self.file_id = f'{sym_string}_{format_age(a)}-{format_age(b)}'
        self.name = f'nihpd_{self.file_id}_nifti'
        self.filename = f'{self.name}.zip'
        self.url = urllib.parse.urljoin(self.url_dir, self.filename)
        if not self.download_root.is_dir():
            download_and_extract_archive(
                self.url,
                download_root=self.download_root,
                filename=self.filename,
            )
            (self.download_root / self.filename).unlink()
            for path in self.download_root.glob('*.nii'):
                compress(path)
                path.unlink()

        try:
            subject_dict = self.get_subject_dict('.nii.gz')
        except FileNotFoundError:  # for backward compatibility
            subject_dict = self.get_subject_dict('.nii')
        super().__init__(subject_dict)

    def get_subject_dict(self, extension):
        root = self.download_root
        subject_dict = {
            't1': ScalarImage(root / f'nihpd_{self.file_id}_t1w{extension}'),
            't2': ScalarImage(root / f'nihpd_{self.file_id}_t2w{extension}'),
            'pd': ScalarImage(root / f'nihpd_{self.file_id}_pdw{extension}'),
            'mask': LabelMap(root / f'nihpd_{self.file_id}_mask{extension}'),
        }
        return subject_dict
