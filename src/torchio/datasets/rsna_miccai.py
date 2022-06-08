import csv
import warnings
from pathlib import Path
from typing import List, Sequence

from ..typing import TypePath
from .. import SubjectsDataset, Subject, ScalarImage


class RSNAMICCAI(SubjectsDataset):
    """RSNA-MICCAI Brain Tumor Radiogenomic Classification challenge dataset.

    This is a helper class for the dataset used in the
    `RSNA-MICCAI Brain Tumor Radiogenomic Classification challenge`_ hosted on
    `kaggle <https://www.kaggle.com/>`_. The dataset must be downloaded before
    instantiating this class (as opposed to, e.g., :class:`torchio.datasets.IXI`).

    This `kaggle kernel <https://www.kaggle.com/fepegar/preprocessing-mri-with-torchio/>`_
    includes a usage example including preprocessing of all the scans.

    If you reference or use the dataset in any form, include the following
    citation:

    U.Baid, et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor
    Segmentation and Radiogenomic Classification", arXiv:2107.02314, 2021.

    Args:
        root_dir: Directory containing the dataset (``train`` directory,
            ``test`` directory, etc.).
        train: If ``True``, the ``train`` set will be used. Otherwise the
            ``test`` set will be used.
        ignore_empty: If ``True``, the three subjects flagged as "presenting
            issues" (empty images) by the challenge organizers will be ignored.
            The subject IDs are ``00109``, ``00123`` and ``00709``.

    Example:
        >>> import torchio as tio
        >>> from subprocess import call
        >>> call('kaggle competitions download -c rsna-miccai-brain-tumor-radiogenomic-classification'.split())
        >>> root_dir = 'rsna-miccai-brain-tumor-radiogenomic-classification'
        >>> train_set = tio.datasets.RSNAMICCAI(root_dir, train=True)
        >>> test_set = tio.datasets.RSNAMICCAI(root_dir, train=False)
        >>> len(train_set), len(test_set)
        (582, 87)


    .. _RSNA-MICCAI Brain Tumor Radiogenomic Classification challenge: https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification
    """  # noqa: E501
    id_key = 'BraTS21ID'
    label_key = 'MGMT_value'
    bad_subjects = '00109', '00123', '00709'

    def __init__(
            self,
            root_dir: TypePath,
            train: bool = True,
            ignore_empty: bool = True,
            modalities: Sequence[str] = ('T1w', 'T1wCE', 'T2w', 'FLAIR'),
            **kwargs,
            ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        if isinstance(modalities, str):
            modalities = [modalities]
        self.modalities = modalities
        subjects = self._get_subjects(self.root_dir, train, ignore_empty)
        super().__init__(subjects, **kwargs)
        self.train = train

    def _get_subjects(
            self,
            root_dir: Path,
            train: bool,
            ignore_empty: bool,
            ) -> List[Subject]:
        subjects = []
        if train:
            csv_path = root_dir / 'train_labels.csv'
            try:
                with open(csv_path) as csvfile:
                    reader = csv.DictReader(csvfile)
                    labels_dict = {
                        row[self.id_key]: int(row[self.label_key])
                        for row in reader
                    }
            except FileNotFoundError:
                warnings.warn('Labels CSV not found. Ignoring MGMT labels')
                labels_dict = {}
            subjects_dir = root_dir / 'train'
        else:
            subjects_dir = root_dir / 'test'

        for subject_dir in sorted(subjects_dir.iterdir()):
            subject_id = subject_dir.name
            if ignore_empty and subject_id in self.bad_subjects:
                continue
            try:
                int(subject_id)
            except ValueError:
                continue
            images_dict = {self.id_key: subject_dir.name}
            if train and labels_dict:
                images_dict[self.label_key] = labels_dict[subject_id]
            for modality in self.modalities:
                image_dir = subject_dir / modality
                filepaths = list(image_dir.iterdir())
                num_files = len(filepaths)
                path = filepaths[0] if num_files == 1 else image_dir
                images_dict[modality] = ScalarImage(path)
            subject = Subject(images_dict)
            subjects.append(subject)
        return subjects
