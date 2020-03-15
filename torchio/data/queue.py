import random
import warnings
from typing import List, Iterable, Iterator
from itertools import islice
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
from .. import TypeTuple
from .images import ImagesDataset
from .sampler import ImageSampler


class Queue(Dataset):
    r"""Patches queue used for patch-based training.

    Args:
        subjects_dataset: Instance of
            :class:`~torchio.data.images.ImagesDataset`.
        max_length: Maximum number of patches that can be stored in the queue.
            Using a large number means that the queue needs to be filled less
            often, but more RAM is needed to store the patches.
        samples_per_volume: Number of patches to extract from each volume.
            A small number of patches ensures a large variability in the queue,
            but training will be slower.
        patch_size: Tuple of integers :math:`(d, h, w)` to generate patches
            of size :math:`d \times h \times w`.
            If a single number :math:`n` is provided,
            :math:`d = h = w = n`.
        sampler_class: An instance of :class:`~torchio.data.ImageSampler` used
            to define the patches sampling strategy.
        num_workers: Number of subprocesses to use for data loading
            (as in :class:`torch.utils.data.DataLoader`).
            ``0`` means that the data will be loaded in the main process.
        shuffle_subjects: If ``True``, the subjects dataset is shuffled at the
            beginning of each epoch, i.e. when all patches from all subjects
            have been processed.
        shuffle_patches: If ``True``, patches are shuffled after filling the
            queue.
        verbose: If ``True``, some debugging messages are printed.

    .. note:: :attr:`num_workers` refers to the number of workers used to
        load and transform the volumes. Multiprocessing is not needed to pop
        patches from the queue.

    Example:

    >>> form torch.utils.data import DataLoader
    >>> import torchio
    >>> patches_queue = torchio.Queue(
    ...     subjects_dataset=subjects_dataset,  # instance of torchio.ImagesDataset
    ...     max_length=300,
    ...     samples_per_volume=10,
    ...     patch_size=96,
    ...     sampler_class=torchio.sampler.ImageSampler,
    ...     num_workers=4,
    ...     shuffle_subjects=True,
    ...     shuffle_patches=True,
    ... )
    >>> patches_loader = DataLoader(patches_queue, batch_size=4)
    >>> num_epochs = 20
    >>> for epoch_index in range(num_epochs):
    ...     for patches_batch in patches_loader:
    ...         inputs = patches_batch['image_name'][torchio.DATA]
    ...         targets = patches_batch['targets_name'][torchio.DATA]
    ...         logits = model(inputs)  # model is some torch.nn.Module

    """
    def __init__(
            self,
            subjects_dataset: ImagesDataset,
            max_length: int,
            samples_per_volume: int,
            patch_size: TypeTuple,
            sampler_class: ImageSampler,
            num_workers: int = 0,
            shuffle_subjects: bool = True,
            shuffle_patches: bool = True,
            verbose: bool = False,
            ):
        self.subjects_dataset = subjects_dataset
        self.max_length = max_length
        self.shuffle_subjects = shuffle_subjects
        self.shuffle_patches = shuffle_patches
        self.samples_per_volume = samples_per_volume
        self.sampler_class = sampler_class
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.subjects_iterable = self.get_subjects_iterable()
        self.patches_list: List[dict] = []
        self.num_sampled_patches = 0

    def __len__(self):
        return self.iterations_per_epoch

    def __getitem__(self, _):
        # There are probably more elegant ways of doing this
        if not self.patches_list:
            self.print('Patches list is empty.')
            self.fill()
        sample_patch = self.patches_list.pop()
        self.num_sampled_patches += 1
        return sample_patch

    def __repr__(self):
        attributes = [
            f'max_length={self.max_length}',
            f'num_subjects={self.num_subjects}',
            f'num_patches={self.num_patches}',
            f'samples_per_volume={self.samples_per_volume}',
            f'num_sampled_patches={self.num_sampled_patches}',
            f'iterations_per_epoch={self.iterations_per_epoch}',
        ]
        attributes_string = ', '.join(attributes)
        return f'Queue({attributes_string})'

    def print(self, *args):
        if self.verbose:
            print(*args)

    @property
    def num_subjects(self) -> int:
        return len(self.subjects_dataset)

    @property
    def num_patches(self) -> int:
        return len(self.patches_list)

    @property
    def iterations_per_epoch(self) -> int:
        return self.num_subjects * self.samples_per_volume

    def fill(self) -> None:
        assert self.sampler_class is not None
        assert self.patch_size is not None
        if self.max_length % self.samples_per_volume != 0:
            message = (
                f'Queue length ({self.max_length})'
                ' not divisible by the number of'
                f' patches per volume ({self.samples_per_volume})'
            )
            warnings.warn(message)

        # If there are e.g. 4 subjects and 1 sample per volume and max_length
        # is 6, we just need to load 4 subjects, not 6
        max_num_subjects_for_queue = self.max_length // self.samples_per_volume
        num_subjects_for_queue = min(
            self.num_subjects, max_num_subjects_for_queue)

        self.print(f'Filling queue from {num_subjects_for_queue} subjects...')
        if self.verbose:
            iterable = trange(num_subjects_for_queue, leave=False)
        else:
            iterable = range(num_subjects_for_queue)
        for _ in iterable:
            subject_sample = self.get_next_subject_sample()
            sampler = self.sampler_class(subject_sample, self.patch_size)
            samples = [s for s in islice(sampler, self.samples_per_volume)]
            assert isinstance(samples, list)
            self.patches_list.extend(samples)
        if self.shuffle_patches:
            random.shuffle(self.patches_list)

    def get_next_subject_sample(self) -> dict:
        # A StopIteration exception is expected when the queue is empty
        try:
            subject_sample = next(self.subjects_iterable)
        except StopIteration as exception:
            self.print('Queue is empty:', exception)
            self.subjects_iterable = self.get_subjects_iterable()
            subject_sample = next(self.subjects_iterable)
        return subject_sample

    def get_subjects_iterable(self) -> Iterator:
        # I need a DataLoader to handle parallelism
        # But this loader is always expected to yield single subject samples
        self.print(
            '\nCreating subjects loader with', self.num_workers, 'workers')
        subjects_loader = DataLoader(
            self.subjects_dataset,
            num_workers=self.num_workers,
            collate_fn=lambda x: x[0],
            shuffle=self.shuffle_subjects,
        )
        return iter(subjects_loader)
