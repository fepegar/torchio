import abc
from pathlib import Path
from typing import Optional

from ..typing import TypePath
from ..transforms import Transform
from ..download import download_and_extract_archive
from .. import SubjectsDataset, Subject, ScalarImage, LabelMap


class BITE(SubjectsDataset, abc.ABC):
    base_url = 'http://www.bic.mni.mcgill.ca/uploads/Services/'

    def __init__(
            self,
            root: TypePath,
            transform: Optional[Transform] = None,
            download: bool = False,
            **kwargs,
            ):
        root = Path(root).expanduser().absolute()
        if download:
            self._download(root)
        subjects_list = self._get_subjects_list(root)
        self.kwargs = kwargs
        super().__init__(subjects_list, transform=transform, **kwargs)

    def _download(self, root: Path):
        raise NotImplementedError

    def _get_subjects_list(self, root: Path):
        raise NotImplementedError


class BITE3(BITE):
    dirname = 'group3'
    """Pre- and post-resection MR images in BITE.

    *The goal of BITE is to share in vivo medical images of patients wtith
    brain tumors to facilitate the development and validation of new image
    processing algorithms.*

    Please check the `BITE website`_ for more information and
    acknowledgments instructions.

    .. _BITE website: http://nist.mni.mcgill.ca/?page_id=672

    Args:
        root: Root directory to which the dataset will be downloaded.
        transform: An instance of
            :class:`~torchio.transforms.transform.Transform`.
        download: If set to ``True``, will download the data into :attr:`root`.
    """
    def _download(self, root: Path):
        if (root / self.dirname).is_dir():
            return
        root.mkdir(exist_ok=True, parents=True)
        filename = f'{self.dirname}.tar.gz'
        url = self.base_url + filename
        download_and_extract_archive(
            url,
            download_root=root,
            md5='e415b63887c40b727c45552614b44634',
        )
        (root / filename).unlink()  # cleanup

    def _get_subjects_list(self, root: Path):
        subjects_dir = root / self.dirname
        subjects = []
        for i in range(1, 15):
            if i == 13:
                continue  # no MRI for this subject
            subject_id = f'{i:02d}'
            subject_dir = subjects_dir / subject_id
            preop_path = subject_dir / f'{subject_id}_preop_mri.mnc'
            postop_path = subject_dir / f'{subject_id}_postop_mri.mnc'
            images_dict = {}
            images_dict['preop'] = ScalarImage(preop_path)
            images_dict['postop'] = ScalarImage(postop_path)
            for fp in subject_dir.glob('*tumor*'):
                images_dict[fp.stem[3:]] = LabelMap(fp)
            subject = Subject(images_dict)
            subjects.append(subject)
        return subjects
