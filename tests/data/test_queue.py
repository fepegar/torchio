from torch.utils.data import DataLoader
from torchio.data import UniformSampler
from torchio import SubjectsDataset, Queue, DATA
from torchio.utils import create_dummy_dataset
from ..utils import TorchioTestCase


class TestQueue(TorchioTestCase):
    """Tests for `queue` module."""
    def setUp(self):
        super().setUp()
        self.subjects_list = create_dummy_dataset(
            num_images=10,
            size_range=(10, 20),
            directory=self.dir,
            suffix='.nii',
            force=False,
        )

    def run_queue(self, num_workers, **kwargs):
        subjects_dataset = SubjectsDataset(self.subjects_list)
        patch_size = 10
        sampler = UniformSampler(patch_size)
        queue_dataset = Queue(
            subjects_dataset,
            max_length=6,
            samples_per_volume=2,
            sampler=sampler,
            **kwargs,
        )
        _ = str(queue_dataset)
        batch_loader = DataLoader(queue_dataset, batch_size=4)
        for batch in batch_loader:
            _ = batch['one_modality'][DATA]
            _ = batch['segmentation'][DATA]

    def test_queue(self):
        self.run_queue(num_workers=0)

    def test_queue_multiprocessing(self):
        self.run_queue(num_workers=2)

    def test_queue_no_start_background(self):
        self.run_queue(num_workers=0, start_background=False)
