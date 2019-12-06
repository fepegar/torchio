import warnings
from itertools import islice
from torch.utils.data import Dataset, DataLoader


class Queue(Dataset):
    def __init__(
            self,
            subjects_dataset,
            max_length,
            samples_per_volume,
            patch_size,
            sampler_class,
            num_workers=0,
            shuffle_dataset=True,
            verbose=False,
            ):
        self.subjects_dataset = subjects_dataset
        self.max_length = max_length
        self.shuffle_dataset = shuffle_dataset
        self.samples_per_volume = samples_per_volume
        self.sampler_class = sampler_class
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.subjects_iterable = self.get_subjects_iterable()
        self.patches_list = []

    def __len__(self):
        return self.iterations_per_epoch

    def __getitem__(self, _):
        """
        There are probably more elegant ways of doing this
        """
        if not self.patches_list:
            self.fill()
        sample_patch = self.patches_list.pop()
        print(sample_patch['image'].shape)
        return sample_patch

    def __repr__(self):
        attributes = [
            f'max_length={self.max_length}',
            f'num_subjects={self.num_subjects}',
            f'num_patches={self.num_patches}',
            f'iterations_per_epoch={self.iterations_per_epoch}',
        ]
        attributes_string = ', '.join(attributes)
        return f'Queue({attributes_string})'

    @property
    def num_subjects(self):
        return len(self.subjects_dataset)

    @property
    def num_patches(self):
        return len(self.patches_list)

    @property
    def iterations_per_epoch(self):
        return self.num_subjects * self.samples_per_volume

    def fill(self):
        assert self.sampler_class is not None
        assert self.patch_size is not None
        if self.max_length % self.samples_per_volume != 0:
            message = (
                f'Samples per volume ({self.samples_per_volume})'
                f' not divisible by max length ({self.max_length})'
            )
            warnings.warn(message)
        num_subjects_for_queue = self.max_length // self.samples_per_volume
        if self.verbose:
            print(f'Filling queue from {num_subjects_for_queue} subjects...')
        for _ in range(num_subjects_for_queue):
            subject_sample = self.get_next_subject_sample()
            sampler = self.sampler_class(subject_sample, self.patch_size)
            samples = [s for s in islice(sampler, self.samples_per_volume)]
            assert isinstance(samples, list)
            self.patches_list.extend(samples)

    def get_next_subject_sample(self):
        try:
            subject_batch = next(self.subjects_iterable)
        except StopIteration:
            self.subjects_iterable = self.get_subjects_iterable()
            subject_batch = next(self.subjects_iterable)
        subject_sample = self.squeeze_batch(subject_batch)
        message = (
            "subject_sample['image'] should have 4 dimensions,"
            f" but has shape {subject_sample['image'].shape}"
        )
        assert subject_sample['image'].ndim == 4, message
        return subject_sample

    @staticmethod
    def squeeze_batch(batch, idx=0):
        for key, value in batch.items():
            batch[key] = value[idx]
        return batch

    def get_subjects_iterable(self):
        """
        I want a DataLoader to handle parallelism
        """
        if self.verbose:
            print(
                'Creating subjects loader with num_workers', self.num_workers)
        loader = DataLoader(
            self.subjects_dataset,
            shuffle=self.shuffle_dataset,
            num_workers=self.num_workers,
        )
        return iter(loader)
