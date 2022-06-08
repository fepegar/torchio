import urllib.parse
import torch
from ...utils import get_torchio_cache_dir, compress
from ...download import download_and_extract_archive
from ... import ScalarImage, LabelMap
from .mni import SubjectMNI


class ICBM2009CNonlinearSymmetric(SubjectMNI):
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
        >>> import torchio as tio
        >>> icbm = tio.datasets.ICBM2009CNonlinearSymmetric()
        >>> icbm
        ICBM2009CNonlinearSymmetric(Keys: ('t1', 'eyes', 'face', 'brain', 't2', 'pd', 'tissues'); images: 7)
        >>> icbm = tio.datasets.ICBM2009CNonlinearSymmetric(load_4d_tissues=False)
        >>> icbm
        ICBM2009CNonlinearSymmetric(Keys: ('t1', 'eyes', 'face', 'brain', 't2', 'pd', 'gm', 'wm', 'csf'); images: 9)

    """  # noqa: E501
    def __init__(self, load_4d_tissues: bool = True):
        self.name = 'mni_icbm152_nlin_sym_09c_nifti'
        self.url_base = 'http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/'
        self.filename = f'{self.name}.zip'
        self.url = urllib.parse.urljoin(self.url_base, self.filename)
        download_root = get_torchio_cache_dir() / self.name
        if not download_root.is_dir():
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
            gm.set_data(torch.cat((gm.data, wm.data, csf.data)))
            gm.save(tissues_path)

        for fp in files_dir.glob('*.nii'):
            compress(fp, fp.with_suffix('.nii.gz'))
            fp.unlink()

        subject_dict = {
            't1': ScalarImage(f'{p}_t1_{m}{s}'),
            'eyes': LabelMap(f'{p}_t1_{m}_eye_mask{s}'),
            'face': LabelMap(f'{p}_t1_{m}_face_mask{s}'),
            'brain': LabelMap(f'{p}_t1_{m}_mask{s}'),
            't2': ScalarImage(f'{p}_t2_{m}{s}'),
            'pd': ScalarImage(f'{p}_csf_{m}{s}'),
        }
        if load_4d_tissues:
            subject_dict['tissues'] = LabelMap(
                tissues_path, channels_last=True)
        else:
            subject_dict['gm'] = LabelMap(f'{p}_gm_{m}{s}')
            subject_dict['wm'] = LabelMap(f'{p}_wm_{m}{s}')
            subject_dict['csf'] = LabelMap(f'{p}_csf_{m}{s}')

        super().__init__(subject_dict)
