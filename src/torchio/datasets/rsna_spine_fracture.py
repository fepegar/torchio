from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from types import ModuleType

from .. import LabelMap
from .. import ScalarImage
from .. import Subject
from .. import SubjectsDataset
from ..typing import TypePath
from ..utils import normalize_path


TypeBoxes = List[Dict[str, Union[str, float, int]]]


class RSNACervicalSpineFracture(SubjectsDataset):
    """RSNA 2022 Cervical Spine Fracture Detection dataset.

    This is a helper class for the dataset used in the
    `RSNA 2022 Cervical Spine Fracture Detection`_ hosted on
    `kaggle <https://www.kaggle.com/>`_. The dataset must be downloaded before
    instantiating this class.

    .. _RSNA 2022 Cervical Spine Fracture Detection: https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/overview/evaluation
    """  # noqa: E501

    UID = 'StudyInstanceUID'

    def __init__(
        self,
        root_dir: TypePath,
        add_segmentations: bool = False,
        add_bounding_boxes: bool = False,
        **kwargs,
    ):
        self.root_dir = normalize_path(root_dir)
        subjects = self._get_subjects(
            add_segmentations,
            add_bounding_boxes,
        )
        super().__init__(subjects, **kwargs)

    @staticmethod
    def _get_image_dirs_dict(images_dir: Path) -> Dict[str, Path]:
        dirs_dict = {}
        for dicom_dir in sorted(images_dir.iterdir()):
            dirs_dict[dicom_dir.name] = dicom_dir
        return dirs_dict

    @staticmethod
    def _get_segs_paths_dict(segs_dir: Path) -> Dict[str, Path]:
        paths_dict = {}
        for image_path in sorted(segs_dir.iterdir()):
            key = image_path.name.replace('.gz', '').replace('.nii', '')
            paths_dict[key] = image_path
        return paths_dict

    def _get_subjects(
        self,
        add_segmentations: bool,
        add_bounding_boxes: bool,
    ) -> List[Subject]:
        subjects = []
        pd = get_pandas()
        from tqdm.auto import tqdm

        split_name = 'train'
        images_dirname = f'{split_name}_images'
        images_dir = self.root_dir / images_dirname
        image_dirs_dict = self._get_image_dirs_dict(images_dir)

        segmentations_dir = self.root_dir / 'segmentations'
        seg_paths_dict = self._get_segs_paths_dict(segmentations_dir)

        bboxes_path = self.root_dir / 'train_bounding_boxes.csv'
        bounding_boxes_df = pd.read_csv(bboxes_path)
        grouped_boxes = bounding_boxes_df.groupby(self.UID)

        df = pd.read_csv(self.root_dir / f'{split_name}.csv')

        for _, row in tqdm(list(df.iterrows())):
            uid = row[self.UID]
            image_dir = image_dirs_dict[uid]
            seg_path = None
            if add_segmentations:
                seg_path = seg_paths_dict.get(uid, None)
            boxes = []
            if add_bounding_boxes:
                try:
                    boxes_df = grouped_boxes.get_group(uid)
                    boxes = [dict(row) for _, row in boxes_df.iterrows()]
                except KeyError:
                    pass
            subject = self._get_subject(
                dict(row),
                image_dir,
                seg_path,
                boxes,
            )
            subjects.append(subject)
        return subjects

    @staticmethod
    def _filter_list(iterable: List[Path], target: str):
        def _filter(path: Path):
            if path.is_dir():
                return target == path.name
            else:
                name = path.name.replace('.gz', '').replace('.nii', '')
                return target == name
        found = list(filter(_filter, iterable))
        if found:
            assert len(found) == 1
            result = found[0]
        else:
            result = None
        return result

    def _get_subject(
        self,
        csv_row_dict: Dict[str, Union[str, int]],
        image_dir: Path,
        seg_path: Optional[Path],
        boxes: TypeBoxes,
    ) -> Subject:
        subject_dict: Dict[str, Any] = {}
        subject_dict.update(csv_row_dict)
        subject_dict['ct'] = ScalarImage(image_dir)
        if seg_path is not None:
            subject_dict['seg'] = LabelMap(seg_path)
        if boxes:
            subject_dict['boxes'] = boxes
        return Subject(**subject_dict)


def get_pandas() -> ModuleType:
    try:
        import pandas
        return pandas
    except ImportError as e:
        message = (
            'Pandas is required for this operation.'
            ' Install pandas with "pip install pandas" and try again'
        )
        raise ImportError(message) from e
