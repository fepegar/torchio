import urllib.parse
import torch
from torchvision.datasets.utils import download_and_extract_archive
from ...utils import get_torchio_cache_dir, compress
from ... import ScalarImage, LabelMap, DATA
from .mni import SubjectMNI


class ICBM2009CNonlinearSymmetryc(SubjectMNI):
    r"""ICBM template.

    More information can be found in the `website
    <http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009>`_.

    .. image:: http://www.bic.mni.mcgill.ca/uploads/ServicesAtlases/mni_icbm152_sym_09c_small.jpg
        :alt: ICBM 2009c Nonlinear Symmetric

    Args:
        load_4d_tissues: If ``True``, the tissue probability maps will be loaded
            together into a 4D image. Otherwise, they will be loaded into
            independent images.

    Example:
        >>> import torchio
        >>> icbm = torchio.datasets.ICBM2009CNonlinearSymmetryc()
        >>> icbm
        ICBM2009CNonlinearSymmetryc(Keys: ('t1', 'eyes', 'face', 'brain', 't2', 'pd', 'tissues'); images: 7)
        >>> icbm = torchio.datasets.ICBM2009CNonlinearSymmetryc(load_4d_tissues=False)
        >>> icbm
        ICBM2009CNonlinearSymmetryc(Keys: ('t1', 'eyes', 'face', 'brain', 't2', 'pd', 'gm', 'wm', 'csf'); images: 9)

    """
    def __init__(self, load_4d_tissues: bool = True):
        self.name = 'mni_icbm152_nlin_sym_09c_nifti'
        self.url_base = 'http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/'
        self.filename = f'{self.name}.zip'
        self.url = urllib.parse.urljoin(self.url_base, self.filename)
        download_root = get_torchio_cache_dir() / self.name
        if download_root.is_dir():
            print(f'Using cache found in {download_root}')
        else:
            download_and_extract_archive(
                self.url,
                download_root=download_root,
                filename=self.filename,
                remove_finished=True,
            )

        files_dir = download_root / 'mni_icbm152_nlin_sym_09c'

        p = files_dir / 'mni_icbm152'
        m = 'tal_nlin_sym_09c'
        s = '.nii.gz'

        tissues_path = files_dir / f'{p}_tissues_{m}.nii.gz'
        if not tissues_path.is_file():
            gm = LabelMap(f'{p}_gm_{m}.nii')
            wm = LabelMap(f'{p}_wm_{m}.nii')
            csf = LabelMap(f'{p}_csf_{m}.nii')
            gm[DATA] = torch.cat((gm[DATA], wm[DATA], csf[DATA]))
            gm.save(tissues_path)

        for fp in files_dir.glob('*.nii'):
            compress(fp, fp.with_suffix('.nii.gz'))
            fp.unlink()

        subject_dict = dict(
            t1=ScalarImage(f'{p}_t1_{m}{s}'),
            eyes=LabelMap(f'{p}_t1_{m}_eye_mask{s}'),
            face=LabelMap(f'{p}_t1_{m}_face_mask{s}'),
            brain=LabelMap(f'{p}_t1_{m}_mask{s}'),
            t2=ScalarImage(f'{p}_t2_{m}{s}'),
            pd=ScalarImage(f'{p}_csf_{m}{s}'),
        )
        if load_4d_tissues:
            subject_dict['tissues'] = LabelMap(tissues_path)
        else:
            subject_dict['gm'] = LabelMap(f'{p}_gm_{m}{s}')
            subject_dict['wm'] = LabelMap(f'{p}_wm_{m}{s}')
            subject_dict['csf'] = LabelMap(f'{p}_csf_{m}{s}')

        super().__init__(subject_dict)
