from torch.utils.data import DataLoader
from torchio.transforms import RandomElasticDeformation
from ..utils import TorchioTestCase


class TestCollate(TorchioTestCase):
    def test_collate(self):
        # Keys missing in one of the samples will not be present in the batch
        # This is relevant for the case in which a transform is applied to some
        # samples only, according to its probability (p argument)
        transform_no = RandomElasticDeformation(p=0, max_displacement=1)
        transform_yes = RandomElasticDeformation(p=1, max_displacement=1)
        sample_no = transform_no(self.sample)
        sample_yes = transform_yes(self.sample)
        data = sample_no, sample_yes

        class Dataset:
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                return self.data[index]

        loader = DataLoader(Dataset(data), batch_size=2)
        next(iter(loader))
