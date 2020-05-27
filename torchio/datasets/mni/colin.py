import urllib.parse
from torchvision.datasets.utils import download_and_extract_archive
from ...utils import get_torchio_cache_dir
from ... import Image, LABEL
from .mni import SubjectMNI


class Colin27(SubjectMNI):
    """Colin27 MNI template.

    Arguments:
        version: Template year. It can ``1998`` or ``2008``.
    """
    def __init__(self, version=1998):
        if version not in (1998, 2008):
            raise ValueError(f'Version must be 1998 or 2008, not "{version}"')
        self.name = f'mni_colin27_{version}_nifti'
        self.url_dir = urllib.parse.urljoin(self.url_base, 'colin27/')
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

        if version == 1998:
            t1, head, mask = [
                download_root / f'colin27_t1_tal_lin{suffix}.nii'
                for suffix in ('', '_headmask', '_mask')
            ]
            super().__init__(
                t1=Image(t1),
                head=Image(head, type=LABEL),
                brain=Image(mask, type=LABEL),
            )
        elif version == 2008:
            t1, t2, pd, label = [
                download_root / f'colin27_{name}_tal_hires.nii'
                for name in ('t1', 't2', 'pd', 'cls')
            ]
            super().__init__(
                t1=Image(t1),
                t2=Image(t2),
                pd=Image(pd),
                cls=Image(label, type=LABEL),
            )
