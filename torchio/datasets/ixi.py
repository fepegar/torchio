"""
The `Information eXtraction from Images (IXI)`_
dataset contains "nearly 600 MR images from normal, healthy subjects",
including "T1, T2 and PD-weighted images,
MRA images and Diffusion-weighted images (15 directions)".


.. note ::
    This data is made available under the
    Creative Commons CC BY-SA 3.0 license.
    If you use it please acknowledge the source of the IXI data, e.g.
    `the IXI website <https://brain-development.org/ixi-dataset/>`_.

.. _Information eXtraction from Images (IXI): https://brain-development.org/ixi-dataset/
"""

# Adapted from
# https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST


import shutil
from pathlib import Path
from typing import Optional, Sequence
from tempfile import NamedTemporaryFile
from torchvision.datasets.utils import download_and_extract_archive
from ..transforms import Transform
from .. import ImagesDataset, Subject, Image, INTENSITY, LABEL, TypePath


class IXI(ImagesDataset):
    """
    Full IXI dataset.

    Args:
        root: Root directory to which the dataset will be downloaded.
        transform: An instance of
            :class:`~torchio.transforms.transform.Transform`.
        download: If set to ``True``, will download the data into :attr:`root`.
        modalities: List of modalities to be downloaded. They must be in
            ``('T1', 'T2', 'PD', 'MRA', 'DTI')``.

    .. warning:: The size of this dataset is multiple GB.
        If you set :attr:`download` to ``True``, it will take some time
        to be downloaded if it is not already present.

    Example::

        >>> import torchio
        >>> transforms = [
        ...     torchio.ToCanonical(),  # to RAS
        ...     torchio.Resample((1, 1, 1)),  # to 1 mm iso
        ... ]
        >>> ixi_dataset = torchio.datasets.IXI(
        ...     'path/to/ixi_root/',
        ...     modalities=('T1', 'T2'),
        ...     transform=torchio.Compose(transforms),
        ...     download=True,
        ... )
        >>> print('Number of subjects in dataset:', len(ixi_dataset))  # 577
        >>> sample_subject = ixi_dataset[0]
        >>> print('Keys in subject sample:', tuple(sample_subject.keys()))  # ('T1', 'T2')
        >>> print('Shape of T1 data:', sample_subject['T1'].shape)  # [1, 180, 268, 268]
        >>> print('Shape of T2 data:', sample_subject['T2'].shape)  # [1, 241, 257, 188]
    """

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
            modalities: Sequence[str] = ('T1', 'T2'),
            **kwargs,
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
    def _get_subjects_list(root, modalities):
        # The number of files for each modality is not the same
        # E.g. 581 for T1, 578 for T2
        # Let's just use the first modality as reference for now
        # I.e. only subjects with all modalities will be included
        one_modality = modalities[0]
        paths = sglob(root / one_modality, '*.nii.gz')
        subjects = []
        for filepath in paths:
            subject_id = get_subject_id(filepath)
            images_dict = dict(subject_id=subject_id)
            images_dict[one_modality] = Image(filepath, INTENSITY)
            for modality in modalities[1:]:
                globbed = sglob(
                    root / modality, f'{subject_id}-{modality}.nii.gz')
                if globbed:
                    assert len(globbed) == 1
                    images_dict[modality] = Image(globbed[0], INTENSITY)
                else:
                    skip_subject = True
                    break
            else:
                skip_subject = False
            if skip_subject:
                continue
            subjects.append(Subject(**images_dict))
        return subjects

    def _download(self, root, modalities):
        """Download the IXI data if it does not exist already."""

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
    r"""
    This is the dataset used in the `notebook`_.
    It is a tiny version of IXI, containing 566 :math:`T_1`-weighted brain MR
    images and their corresponding brain segmentations,
    all with size :math:`83 \times 44 \times 55`.

    It can be used as a medical image MNIST.

    Args:
        root: Root directory to which the dataset will be downloaded.
        transform: An instance of
            :class:`~torchio.transforms.transform.Transform`.
        download: If set to ``True``, will download the data into :attr:`root`.

    .. _notebook: https://colab.research.google.com/drive/112NTL8uJXzcMw4PQbUvMQN-WHlVwQS3i
    """
    url = 'https://www.dropbox.com/s/ogxjwjxdv5mieah/ixi_tiny.zip?dl=1'
    md5 = 'bfb60f4074283d78622760230bfa1f98'

    def __init__(
            self,
            root: TypePath,
            transform: Optional[Transform] = None,
            download: bool = False,
            **kwargs,
            ):
        root = Path(root)
        if download:
            self._download(root)
        if not root.is_dir():
            message = (
                'Dataset not found.'
                ' You can use download=True to download it'
            )
            raise RuntimeError(message)
        subjects_list = self._get_subjects_list(root)
        super().__init__(subjects_list, transform=transform, **kwargs)

    @staticmethod
    def _get_subjects_list(root):
        image_paths = sglob(root / 'image', '*.nii.gz')
        label_paths = sglob(root / 'label', '*.nii.gz')
        assert image_paths and label_paths

        subjects = []
        for image_path, label_path in zip(image_paths, label_paths):
            subject_id = get_subject_id(image_path)
            subject_dict = {}
            subject_dict['image'] = Image(image_path, INTENSITY)
            subject_dict['label'] = Image(label_path, LABEL)
            subject_dict['subject_id'] = subject_id
            subjects.append(Subject(**subject_dict))
        return subjects

    def _download(self, root):
        """Download the tiny IXI data if it doesn't exist already."""
        if root.is_dir():  # assume it's been downloaded
            print('Root directory for IXITiny found:', root)
            return
        print('Root directory for IXITiny not found:', root)
        print('Downloading...')
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
