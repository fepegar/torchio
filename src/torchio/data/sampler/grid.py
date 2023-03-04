from typing import Generator
from typing import Optional
from typing import Union

import numpy as np

from ...data.subject import Subject
from ...typing import TypeSpatialShape
from ...typing import TypeTripletInt
from ...utils import to_tuple
from .sampler import PatchSampler


class GridSampler(PatchSampler):
    r"""Extract patches across a whole volume.

    Grid samplers are useful to perform inference using all patches from a
    volume. It is often used with a :class:`~torchio.data.GridAggregator`.

    Args:
        subject: Instance of :class:`~torchio.data.Subject`
            from which patches will be extracted.
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided,
            :math:`w = h = d = n`.
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

    Example:

        >>> import torchio as tio
        >>> colin = tio.datasets.Colin27()
        >>> sampler = tio.GridSampler(colin, patch_size=88)
        >>> for i, patch in enumerate(sampler()):
        ...     patch.t1.save(f'patch_{i}.nii.gz')
        ...
        >>> # To figure out the number of patches beforehand:
        >>> sampler = tio.GridSampler(colin, patch_size=88)
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
        subject: Subject,
        patch_size: TypeSpatialShape,
        patch_overlap: TypeSpatialShape = (0, 0, 0),
        padding_mode: Union[str, float, None] = None,
    ):
        super().__init__(patch_size)
        self.patch_overlap = np.array(to_tuple(patch_overlap, length=3))
        self.padding_mode = padding_mode
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

    def __call__(
        self,
        subject: Optional[Subject] = None,
        num_patches: Optional[int] = None,
    ) -> Generator[Subject, None, None]:
        subject = self.subject if subject is None else subject
        return super().__call__(subject, num_patches=num_patches)

    def _pad(self, subject: Subject) -> Subject:
        if self.padding_mode is not None:
            from ...transforms import Pad

            border = self.patch_overlap // 2
            padding = border.repeat(2)
            pad = Pad(padding, padding_mode=self.padding_mode)  # type: ignore[arg-type]  # noqa: B950
            subject = pad(subject)  # type: ignore[assignment]
        return subject

    def _compute_locations(self, subject: Subject):
        sizes = subject.spatial_shape, self.patch_size, self.patch_overlap
        self._parse_sizes(*sizes)  # type: ignore[arg-type]
        return self._get_patches_locations(*sizes)  # type: ignore[arg-type]

    def _generate_patches(  # type: ignore[override]
        self,
        subject: Subject,
    ) -> Generator[Subject, None, None]:
        subject = self._pad(subject)
        sizes = subject.spatial_shape, self.patch_size, self.patch_overlap
        self._parse_sizes(*sizes)  # type: ignore[arg-type]
        locations = self._get_patches_locations(*sizes)  # type: ignore[arg-type]  # noqa: B950
        for location in locations:
            index_ini = location[:3]
            yield self.extract_patch(subject, index_ini)

    @staticmethod
    def _parse_sizes(
        image_size: TypeTripletInt,
        patch_size: TypeTripletInt,
        patch_overlap: TypeTripletInt,
    ) -> None:
        image_size_array = np.array(image_size)
        patch_size_array = np.array(patch_size)
        patch_overlap_array = np.array(patch_overlap)
        if np.any(patch_size_array > image_size_array):
            message = (
                f'Patch size {tuple(patch_size_array)} cannot be'
                f' larger than image size {tuple(image_size_array)}'
            )
            raise ValueError(message)
        if np.any(patch_overlap_array >= patch_size_array):
            message = (
                f'Patch overlap {tuple(patch_overlap_array)} must be smaller'
                f' than patch size {tuple(patch_size_array)}'
            )
            raise ValueError(message)
        if np.any(patch_overlap_array % 2):
            message = (
                'Patch overlap must be a tuple of even integers,'
                f' not {tuple(patch_overlap_array)}'
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
