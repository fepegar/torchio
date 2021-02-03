import csv
from pathlib import Path
from typing import Optional

from ..typing import TypePath
from ..transforms import Transform
from ..download import download_and_extract_archive
from .. import SubjectsDataset, Subject, ScalarImage, LabelMap


class EPISURG(SubjectsDataset):
    """
    `EPISURG`_ is a clinical dataset of :math:`T_1`-weighted MRI from 430
    epileptic patients who underwent resective brain surgery at the National
    Hospital of Neurology and Neurosurgery (Queen Square, London, United
    Kingdom) between 1990 and 2018.

    The dataset comprises 430 postoperative MRI. The corresponding preoperative
    MRI is present for 268 subjects.

    Three human raters segmented the resection cavity on partially overlapping
    subsets of EPISURG.

    If you use this dataset for your research, you agree with the *Data use
    agreement* presented at the EPISURG entry on the `UCL Research Data
    Repository <EPISURG>`_ and you must cite the corresponding publications.

    .. _EPISURG: https://doi.org/10.5522/04/9996158.v1

    Args:
        root: Root directory to which the dataset will be downloaded.
        transform: An instance of
            :class:`~torchio.transforms.transform.Transform`.
        download: If set to ``True``, will download the data into :attr:`root`.

    .. warning:: The size of this dataset is multiple GB.
        If you set :attr:`download` to ``True``, it will take some time
        to be downloaded if it is not already present.
    """

    data_url = 'https://s3-eu-west-1.amazonaws.com/pstorage-ucl-2748466690/26153588/EPISURG.zip'  # noqa: E501
    md5 = '5ec5831a2c6fbfdc8489ba2910a6504b'

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

    @staticmethod
    def _check_exists(root, modalities):
        for modality in modalities:
            modality_dir = root / modality
            if not modality_dir.is_dir():
                exists = False
                break
        else:
            exists = True
        return exists

    @staticmethod
    def _get_subjects_list(root):
        episurg_dir = root / 'EPISURG'
        subjects_dir = episurg_dir / 'subjects'
        csv_path = episurg_dir / 'subjects.csv'
        with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            subjects = []
            for row in reader:
                subject_id = row['Subject']
                subject_dir = subjects_dir / subject_id
                subject_dict = {
                    'subject_id': subject_id,
                    'hemisphere': row['Hemisphere'],
                    'surgery_type': row['Type'],
                }
                preop_dir = subject_dir / 'preop'
                preop_paths = list(preop_dir.glob('*preop*'))
                assert len(preop_paths) <= 1
                if preop_paths:
                    subject_dict['preop_mri'] = ScalarImage(preop_paths[0])
                postop_dir = subject_dir / 'postop'
                postop_path = list(postop_dir.glob('*postop-t1mri*'))[0]
                subject_dict['postop_mri'] = ScalarImage(postop_path)
                for seg_path in postop_dir.glob('*seg*'):
                    seg_id = seg_path.name[-8]
                    subject_dict[f'seg_{seg_id}'] = LabelMap(seg_path)
                subjects.append(Subject(**subject_dict))
        return subjects

    def _download(self, root: Path):
        """Download the EPISURG data if it does not exist already."""
        if (root / 'EPISURG').is_dir():
            return
        root.mkdir(exist_ok=True, parents=True)
        download_and_extract_archive(
            self.data_url,
            download_root=root,
            md5=self.md5,
        )
        (root / 'EPISURG.zip').unlink()  # cleanup

    def _glob_subjects(self, string):
        subjects = []
        for subject in self._subjects:
            for image_name in subject:
                if string in image_name:
                    subjects.append(subject)
                    break
        return subjects

    def _get_labeled_subjects(self):
        return self._glob_subjects('seg')

    def _get_paired_subjects(self):
        return self._glob_subjects('preop')

    def _get_subset(self, subjects):
        dataset = SubjectsDataset(
            subjects,
            transform=self._transform,
            **(self.kwargs),
        )
        return dataset

    def get_labeled(self) -> SubjectsDataset:
        """Get dataset from subjects with manual annotations."""
        return self._get_subset(self._get_labeled_subjects())

    def get_unlabeled(self) -> SubjectsDataset:
        """Get dataset from subjects without manual annotations."""
        subjects = [
            s for s in self._subjects
            if s not in self._get_labeled_subjects()
        ]
        return self._get_subset(subjects)

    def get_paired(self) -> SubjectsDataset:
        """Get dataset from subjects with pre- and post-op MRI."""
        return self._get_subset(self._get_paired_subjects())
