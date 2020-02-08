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
        """There are probably more elegant ways of doing this"""
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
                f'Samples per volume ({self.samples_per_volume})'
                f' not divisible by max length ({self.max_length})'
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
        """A StopIteration exception is expected when the queue is empty"""
        try:
            subject_sample = next(self.subjects_iterable)
        except StopIteration as exception:
            self.print('Queue is empty:', exception)
            self.subjects_iterable = self.get_subjects_iterable()
            subject_sample = next(self.subjects_iterable)
        return subject_sample

    def get_subjects_iterable(self) -> Iterator:
        """
        I need a DataLoader to handle parallelism
        But this loader is always expected to yield single subject samples
        """
        self.print(
            '\nCreating subjects loader with', self.num_workers, 'workers')
        subjects_loader = DataLoader(
            self.subjects_dataset,
            num_workers=self.num_workers,
            collate_fn=lambda x: x[0],
            shuffle=self.shuffle_subjects,
        )
        return iter(subjects_loader)
