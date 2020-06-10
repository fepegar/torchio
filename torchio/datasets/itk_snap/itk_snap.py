import urllib.parse
from torchvision.datasets.utils import download_and_extract_archive
from ...utils import get_torchio_cache_dir
from ... import Subject, Image, LABEL


class SubjectITKSNAP(Subject):
    """ITK-SNAP Image Data Downloads.

    See `the itk-SNAP website <http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.Data>`_
    for more information.
    """
    url_base = 'https://www.nitrc.org/frs/download.php/'

    def __init__(self, name, code):
        self.name = name
        self.url_dir = urllib.parse.urljoin(self.url_base, f'{code}/')
        self.filename = f'{self.name}.zip'
        self.url = urllib.parse.urljoin(self.url_dir, self.filename)
        self.download_root = get_torchio_cache_dir() / self.name
        if self.download_root.is_dir():
            print(f'Using cache found in {self.download_root}')
        else:
            download_and_extract_archive(
                self.url,
                download_root=self.download_root,
                filename=self.filename,
            )
        super().__init__(**self.get_kwargs())

    def get_kwargs(self):
        raise NotImplementedError


class BrainTumor(SubjectITKSNAP):
    def __init__(self):
        super().__init__('braintumor', '6161')

    def get_kwargs(self):
        t1, t1c, t2, flair, seg = [
            self.download_root / self.name / f'BRATS_HG0015_{name}.mha'
            for name in ('T1', 'T1C', 'T2', 'FLAIR', 'truth')
        ]
        return dict(
            t1=Image(t1),
            t1c=Image(t1c),
            t2=Image(t2),
            flair=Image(flair),
            seg=Image(seg, type=LABEL),
        )


class T1T2(SubjectITKSNAP):
    def __init__(self):
        super().__init__('ashs_test', '10983')

    def get_kwargs(self):
        mprage = self.download_root / self.name / 'mprage_3T_bet_dr.nii'
        tse = self.download_root / self.name / 'tse_3t_dr.nii'
        return dict(
            mprage=Image(mprage),
            tse=Image(tse),
        )


class AorticValve(SubjectITKSNAP):
    def __init__(self):
        super().__init__('bav_example', '11021')

    def get_kwargs(self):
        b14, b14_seg, b25, b25_seg = [
            self.download_root / self.name / f'bav_frame_{name}.nii.gz'
            for name in ('14', '14_manseg', '25', '25_manseg')
        ]
        return dict(
            b14=Image(b14),
            b14_seg=Image(b14_seg, type=LABEL),
            b25=Image(b25),
            b25_seg=Image(b25_seg, type=LABEL),
        )
