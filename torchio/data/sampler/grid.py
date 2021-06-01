from typing import Union, Generator, Optional

import numpy as np

from ...utils import to_tuple
from ...data.subject import Subject
from ...typing import TypePatchSize
from ...typing import TypeTripletInt
from .sampler import PatchSampler


class GridSampler(PatchSampler):
    r"""Extract patches across a whole volume.

    Grid samplers are useful to perform inference using all patches from a
    volume. It is often used with a :class:`~torchio.data.GridAggregator`.

    Args:
        subject: Instance of :class:`~torchio.data.Subject`
            from which patches will be extracted. This argument should only be
            used before instantiating a :class:`~torchio.data.GridAggregator`,
            or to precompute the number of patches that would be generated from
            a subject.
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided,
            :math:`w = h = d = n`.
            This argument is mandatory (it is a keyword argument for backward
            compatibility).
        patch_overlap: Tuple of even integers :math:`(w_o, h_o, d_o)`
            specifying the overlap between patches for dense inference. If a
            single number :math:`n` is provided, :math:`w_o = h_o = d_o = n`.
        padding_mode: Same as :attr:`padding_mode` in
            :class:`~torchio.transforms.Pad`. If ``None``, the volume will not
            be padded before sampling and patches at the border will not be
            cropped by the aggregator.
            Otherwise, the volume will be padded with
            :math:`\left(\frac{w_o}{2}, \frac{h_o}{2}, \frac{d_o}{2} \right)`
            on each side before sampling. If the sampler is passed to a
            :class:`~torchio.data.GridAggregator`, it will crop the output
            to its original size.

    Example::

        >>> import torchio as tio
        >>> sampler = tio.GridSampler(patch_size=88)
        >>> colin = tio.datasets.Colin27()
        >>> for i, patch in enumerate(sampler(colin)):
        ...     patch.t1.save(f'patch_{i}.nii.gz')
        ...
        >>> # To figure out the number of patches beforehand:
        >>> sampler = tio.GridSampler(subject=colin, patch_size=88)
        >>> len(sampler)
        8

    .. note:: Adapted from NiftyNet. See `this NiftyNet tutorial
        <https://niftynet.readthedocs.io/en/dev/window_sizes.html>`_ for more
        information about patch based sampling. Note that
        :attr:`patch_overlap` is twice :attr:`border` in NiftyNet
        tutorial.
    """
    def __init__(
            self,
            subject: Optional[Subject] = None,
            patch_size: TypePatchSize = None,
            patch_overlap: TypePatchSize = (0, 0, 0),
            padding_mode: Union[str, float, None] = None,
            ):
        if patch_size is None:
            raise ValueError('A value for patch_size must be given')
        super().__init__(patch_size)
        self.patch_overlap = np.array(to_tuple(patch_overlap, length=3))
        self.padding_mode = padding_mode
        if subject is not None and not isinstance(subject, Subject):
            raise ValueError('The subject argument must be None or Subject')
        self.subject = self._pad(subject)
        self.locations = self._compute_locations(self.subject)

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        # Assume 3D
        location = self.locations[index]
        index_ini = location[:3]
        cropped_subject = self.crop(self.subject, index_ini, self.patch_size)
        return cropped_subject

    def _pad(self, subject: Subject) -> Subject:
        if self.padding_mode is not None:
            from ...transforms import Pad
            border = self.patch_overlap // 2
            padding = border.repeat(2)
            pad = Pad(padding, padding_mode=self.padding_mode)
            subject = pad(subject)
        return subject

    def _compute_locations(self, subject: Subject):
        if subject is None:
            return None
        sizes = subject.spatial_shape, self.patch_size, self.patch_overlap
        self._parse_sizes(*sizes)
        return self._get_patches_locations(*sizes)

    def _generate_patches(
            self,
            subject: Subject,
            ) -> Generator[Subject, None, None]:
        subject = self._pad(subject)
        sizes = subject.spatial_shape, self.patch_size, self.patch_overlap
        self._parse_sizes(*sizes)
        locations = self._get_patches_locations(*sizes)
        for location in locations:
            index_ini = location[:3]
            yield self.extract_patch(subject, index_ini)

    @staticmethod
    def _parse_sizes(
            image_size: TypeTripletInt,
            patch_size: TypeTripletInt,
            patch_overlap: TypeTripletInt,
            ) -> None:
        image_size = np.array(image_size)
        patch_size = np.array(patch_size)
        patch_overlap = np.array(patch_overlap)
        if np.any(patch_size > image_size):
            message = (
                f'Patch size {tuple(patch_size)} cannot be'
                f' larger than image size {tuple(image_size)}'
            )
            raise ValueError(message)
        if np.any(patch_overlap >= patch_size):
            message = (
                f'Patch overlap {tuple(patch_overlap)} must be smaller'
                f' than patch size {tuple(patch_size)}'
            )
            raise ValueError(message)
        if np.any(patch_overlap % 2):
            message = (
                'Patch overlap must be a tuple of even integers,'
                f' not {tuple(patch_overlap)}'
            )
            raise ValueError(message)

    @staticmethod
    def _get_patches_locations(
            image_size: TypeTripletInt,
            patch_size: TypeTripletInt,
            patch_overlap: TypeTripletInt,
            ) -> np.ndarray:
        # Example with image_size 10, patch_size 5, overlap 2:
        # [0 1 2 3 4 5 6 7 8 9]
        # [0 0 0 0 0]
        #       [1 1 1 1 1]
        #           [2 2 2 2 2]
        # Locations:
        # [[0, 5],
        #  [3, 8],
        #  [5, 10]]
        indices = []
        zipped = zip(image_size, patch_size, patch_overlap)
        for im_size_dim, patch_size_dim, patch_overlap_dim in zipped:
            end = im_size_dim + 1 - patch_size_dim
            step = patch_size_dim - patch_overlap_dim
            indices_dim = list(range(0, end, step))
            if indices_dim[-1] != im_size_dim - patch_size_dim:
                indices_dim.append(im_size_dim - patch_size_dim)
            indices.append(indices_dim)
        indices_ini = np.array(np.meshgrid(*indices)).reshape(3, -1).T
        indices_ini = np.unique(indices_ini, axis=0)
        indices_fin = indices_ini + np.array(patch_size)
        locations = np.hstack((indices_ini, indices_fin))
        return np.array(sorted(locations.tolist()))
