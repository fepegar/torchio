"""
Create a TorchIO dataset with the IXI data from
https://brain-development.org/ixi-dataset/

Adapted from
https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST
"""

import shutil
from pathlib import Path
from typing import Optional, Sequence
from tempfile import NamedTemporaryFile
from ..transforms import Transform
from .. import ImagesDataset, Subject, Image, INTENSITY, LABEL, TypePath
from torchvision.datasets.utils import download_and_extract_archive


class IXI(ImagesDataset):
    base_url = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-{modality}.tar'
    md5_dict = {
        'T1': '34901a0593b41dd19c1a1f746eac2d58',
        'T2': 'e3140d78730ecdd32ba92da48c0a9aaa',
        'PD': '88ecd9d1fa33cb4a2278183b42ffd749',
        'MRA': '29be7d2fee3998f978a55a9bdaf3407e',
        'DTI': '636573825b1c8b9e8c78f1877df3ee66',
    }

    def __init__(
            self,
            root: TypePath,
            transform: Optional[Transform] = None,
            download: bool = False,
            modalities: Sequence[str] = ['T1', 'T2'],
            ):
        root = Path(root)
        for modality in modalities:
            if modality not in self.md5_dict:
                message = (
                    f'Modality "{modality}" must be'
                    f' one of {tuple(self.md5_dict.keys())}'
                )
                raise ValueError(message)
        if download:
            self._download(root, modalities)
        if not self._check_exists(root, modalities):
            message = (
                'Dataset not found.'
                ' You can use download=True to download it'
            )
            raise RuntimeError(message)
        subjects_list = self._get_subjects_list(root, modalities)
        super().__init__(subjects_list, transform=transform)

    def _check_exists(self, root, modalities):
        for modality in modalities:
            modality_dir = root / modality
            if not modality_dir.is_dir():
                exists = False
                break
        else:
            exists = True
        return exists

    def _get_subjects_list(self, root, modalities):
        """
        The number of files for each modality is not the same
        E.g. 581 for T1, 578 for T2
        Let's just use the first modality as reference for now
        I.e. only subjects with all modalities will be included
        """
        def sglob(directory, pattern):
            return sorted(list(Path(directory).glob(pattern)))

        def get_subject_id(path):
            return '-'.join(path.name.split('-')[:-1])

        one_modality = modalities[0]
        paths = sglob(root / one_modality, '*.nii.gz')
        subjects = []
        for filepath in paths:
            subject_id = get_subject_id(filepath)
            images = []
            images.append(Image(one_modality, filepath, INTENSITY))
            for modality in modalities[1:]:
                globbed = sglob(
                    root / modality, f'{subject_id}-{modality}.nii.gz')
                if globbed:
                    assert len(globbed) == 1
                    images.append(Image(modality, globbed[0], INTENSITY))
                else:
                    skip_subject = True
                    break
            else:
                skip_subject = False
            if skip_subject:
                continue
            subjects.append(Subject(*images, name=subject_id))
        return subjects

    def _download(self, root, modalities):
        """Download the IXI data if it doesn't exist already."""

        for modality in modalities:
            modality_dir = root / modality
            if modality_dir.is_dir():
                continue
            modality_dir.mkdir(exist_ok=True, parents=True)

            # download files
            url = self.base_url.format(modality=modality)
            md5 = self.md5_dict[modality]

            with NamedTemporaryFile(suffix='.tar') as f:
                download_and_extract_archive(
                    url,
                    download_root=modality_dir,
                    filename=f.name,
                    md5=md5,
                )


class IXITiny(ImagesDataset):
    url = 'https://www.dropbox.com/s/ogxjwjxdv5mieah/ixi_tiny.zip?dl=1'
    md5 = 'bfb60f4074283d78622760230bfa1f98'

    def __init__(
            self,
            root: TypePath,
            transform: Optional[Transform] = None,
            download: bool = False,
            ):
        root = Path(root)
        if download:
            self._download(root)
        if not self._check_exists(root):
            message = (
                'Dataset not found.'
                ' You can use download=True to download it'
            )
            raise RuntimeError(message)
        subjects_list = self._get_subjects_list(root)
        super().__init__(subjects_list, transform=transform)

    def _check_exists(self, root):
        return root.is_dir()

    def _get_subjects_list(self, root):
        image_paths = sglob(root / 'image', '*.nii.gz')
        label_paths = sglob(root / 'label', '*.nii.gz')

        subjects = []
        for image_path, label_path in zip(image_paths, label_paths):
            subject_id = get_subject_id(image_path)
            images = []
            images.append(Image('image', image_path, INTENSITY))
            images.append(Image('label', label_path, LABEL))
            subjects.append(Subject(*images, name=subject_id))
        return subjects

    def _download(self, root):
        """Download the tiny IXI data if it doesn't exist already."""

        with NamedTemporaryFile(suffix='.zip') as f:
            download_and_extract_archive(
                self.url,
                download_root=root,
                filename=f.name,
                md5=self.md5,
            )
        ixi_tiny_dir = root / 'ixi_tiny'
        (ixi_tiny_dir / 'image').rename(root / 'image')
        (ixi_tiny_dir / 'label').rename(root / 'label')
        shutil.rmtree(ixi_tiny_dir)


def sglob(directory, pattern):
    return sorted(list(Path(directory).glob(pattern)))

def get_subject_id(path):
    return '-'.join(path.name.split('-')[:-1])
