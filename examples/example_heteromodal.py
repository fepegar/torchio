"""
This is an example of a very particular case in which some modalities might be
missing for some of the subjects, as in

    Dorent et al. 2019, Hetero-Modal Variational Encoder-Decoder
    for Joint Modality Completion and Segmentation

"""

import torch.nn as nn
from torch.utils.data import DataLoader

import torchio
from torchio import Image, Subject, ImagesDataset, Queue
from torchio.data import UniformSampler

def main():
    # Define training and patches sampling parameters
    num_epochs = 20
    patch_size = 128
    queue_length = 100
    samples_per_volume = 5
    batch_size = 2

    # Populate a list with images
    one_subject = Subject(
        T1=Image('../BRATS2018_crop_renamed/LGG75_T1.nii.gz', torchio.INTENSITY),
        T2=Image('../BRATS2018_crop_renamed/LGG75_T2.nii.gz', torchio.INTENSITY),
        label=Image('../BRATS2018_crop_renamed/LGG75_Label.nii.gz', torchio.LABEL),
    )

    # This subject doesn't have a T2 MRI!
    another_subject = Subject(
        T1=Image('../BRATS2018_crop_renamed/LGG74_T1.nii.gz', torchio.INTENSITY),
        label=Image('../BRATS2018_crop_renamed/LGG74_Label.nii.gz', torchio.LABEL),
    )

    subjects = [
        one_subject,
        another_subject,
    ]

    subjects_dataset = ImagesDataset(subjects)
    queue_dataset = Queue(
        subjects_dataset,
        queue_length,
        samples_per_volume,
        UniformSampler(patch_size),
    )

    # This collate_fn is needed in the case of missing modalities
    # In this case, the batch will be composed by a *list* of samples instead of
    # the typical Python dictionary that is collated by default in Pytorch
    batch_loader = DataLoader(
        queue_dataset,
        batch_size=batch_size,
        collate_fn=lambda x: x,
    )

    # Mock PyTorch model
    model = nn.Identity()

    for epoch_index in range(num_epochs):
        for batch in batch_loader:  # batch is a *list* here, not a dictionary
            logits = model(batch)
            print([batch[idx].keys() for idx in range(batch_size)])
    print()


if __name__ == "__main__":
    main()
