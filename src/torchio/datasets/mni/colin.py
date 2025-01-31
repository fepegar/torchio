import urllib.parse

from ...data import LabelMap
from ...data import ScalarImage
from ...download import download_and_extract_archive
from ...utils import compress
from .mni import SubjectMNI

TISSUES_2008 = {
    1: 'Cerebro-spinal fluid',
    2: 'Gray Matter',
    3: 'White Matter',
    4: 'Fat',
    5: 'Muscles',
    6: 'Skin and Muscles',
    7: 'Skull',
    9: 'Fat 2',
    10: 'Dura',
    11: 'Marrow',
    12: 'Vessels',
}


class Colin27(SubjectMNI):
    NAME_TO_LABEL = {name: label for label, name in TISSUES_2008.items()}
    r"""Colin27 MNI template.

    More information can be found in the website of the
    `1998 <https://nist.mni.mcgill.ca/colin-27-average-brain/>`_ and
    `2008 <http://www.bic.mni.mcgill.ca/ServicesAtlases/Colin27Highres>`_
    versions.

    .. image:: http://www.bic.mni.mcgill.ca/uploads/ServicesAtlases/mni_colin27_2008.jpg
        :alt: MNI Colin 27 2008 version

    Arguments:
        version: Template year. It can be ``1998`` or ``2008``.

    .. warning:: The resolution of the ``2008`` version is quite high. The
        subject instance will contain four images of size
        :math:`362 \times 434 \times 362`, therefore applying a transform to
        it might take longer than expected.

    Example:
        >>> import torchio as tio
        >>> colin_1998 = tio.datasets.Colin27(version=1998)
        >>> colin_1998
        Colin27(Keys: ('t1', 'head', 'brain'); images: 3)
        >>> colin_1998.load()
        >>> colin_1998.t1
        ScalarImage(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; memory: 27.1 MiB; type: intensity)
        >>>
        >>> colin_2008 = tio.datasets.Colin27(version=2008)
        >>> colin_2008
        Colin27(Keys: ('t1', 't2', 'pd', 'cls'); images: 4)
        >>> colin_2008.load()
        >>> colin_2008.t1
        ScalarImage(shape: (1, 362, 434, 362); spacing: (0.50, 0.50, 0.50); orientation: RAS+; memory: 217.0 MiB; type: intensity)
    """

    def __init__(self, version=1998):
        if version not in (1998, 2008):
            raise ValueError(f'Version must be 1998 or 2008, not "{version}"')
        self.version = version
        self.name = f'mni_colin27_{version}_nifti'
        self.url_dir = urllib.parse.urljoin(self.url_base, 'colin27/')
        self.filename = f'{self.name}.zip'
        self.url = urllib.parse.urljoin(self.url_dir, self.filename)
        if not self.download_root.is_dir():
            download_and_extract_archive(
                self.url,
                download_root=self.download_root,
                filename=self.filename,
            )

            # Fix label map (https://github.com/TorchIO-project/torchio/issues/220)
            if version == 2008:
                path = self.download_root / 'colin27_cls_tal_hires.nii'
                cls_image = LabelMap(path)
                cls_image.set_data(cls_image.data.round().byte())
                cls_image.save(path)

            (self.download_root / self.filename).unlink()
            for path in self.download_root.glob('*.nii'):
                compress(path)
                path.unlink()

        try:
            subject_dict = self.get_subject_dict(
                self.download_root,
                extension='.nii.gz',
            )
        except FileNotFoundError:  # for backward compatibility
            subject_dict = self.get_subject_dict(
                self.download_root,
                extension='.nii',
            )
        super().__init__(subject_dict)

    def get_subject_dict(self, download_root, extension):
        if self.version == 1998:
            subject_dict = Colin1998.get_subject_dict(download_root, extension)
        elif self.version == 2008:
            subject_dict = Colin2008.get_subject_dict(download_root, extension)
        return subject_dict


class Colin1998:
    @staticmethod
    def get_subject_dict(download_root, extension):
        t1, head, mask = (
            download_root / f'colin27_t1_tal_lin{suffix}{extension}'
            for suffix in ('', '_headmask', '_mask')
        )
        subject_dict = {
            't1': ScalarImage(t1),
            'head': LabelMap(head),
            'brain': LabelMap(mask),
        }
        return subject_dict


class Colin2008:
    @staticmethod
    def get_subject_dict(download_root, extension):
        t1, t2, pd, label = (
            download_root / f'colin27_{name}_tal_hires{extension}'
            for name in ('t1', 't2', 'pd', 'cls')
        )
        subject_dict = {
            't1': ScalarImage(t1),
            't2': ScalarImage(t2),
            'pd': ScalarImage(pd),
            'cls': LabelMap(label, labels=TISSUES_2008),
        }
        return subject_dict
