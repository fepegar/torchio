from torch.utils.data import DataLoader
import torchio as tio
from ..utils import TorchioTestCase


class TestCollate(TorchioTestCase):

    def get_heterogeneous_dataset(self):
        # Keys missing in one of the samples will not be present in the batch
        # This is relevant for the case in which a transform is applied to some
        # samples only, according to its probability (p argument)
        transform_no = tio.RandomElasticDeformation(p=0, max_displacement=1)
        transform_yes = tio.RandomElasticDeformation(p=1, max_displacement=1)
        sample_no = transform_no(self.sample_subject)
        sample_yes = transform_yes(self.sample_subject)
        data = sample_no, sample_yes

        class Dataset:

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                return self.data[index]

        return Dataset(data)

    def test_collate(self):
        loader = DataLoader(self.get_heterogeneous_dataset(), batch_size=2)
        next(iter(loader))

    def test_history_collate(self):
        loader = DataLoader(
            self.get_heterogeneous_dataset(),
            batch_size=4,
            collate_fn=tio.utils.history_collate,
        )
        batch = next(iter(loader))
        empty_history, one_history = batch['history']
        assert not empty_history
        assert len(one_history) == 1
