"""This is an example of a very particular case in which some modalities might
be missing for some of the subjects, as in.

Dorent et al. 2019, Hetero-Modal Variational Encoder-Decoder for Joint
Modality Completion and Segmentation
"""

import logging

import torch.nn as nn

import torchio as tio
from torchio import LabelMap
from torchio import Queue
from torchio import ScalarImage
from torchio import Subject
from torchio import SubjectsDataset
from torchio.data import UniformSampler


def main():
    # Define training and patches sampling parameters
    num_epochs = 20
    patch_size = 128
    queue_length = 100
    patches_per_volume = 5
    batch_size = 2

    # Populate a list with images
    one_subject = Subject(
        T1=ScalarImage('../BRATS2018_crop_renamed/LGG75_T1.nii.gz'),
        T2=ScalarImage('../BRATS2018_crop_renamed/LGG75_T2.nii.gz'),
        label=LabelMap('../BRATS2018_crop_renamed/LGG75_Label.nii.gz'),
    )

    # This subject doesn't have a T2 MRI!
    another_subject = Subject(
        T1=ScalarImage('../BRATS2018_crop_renamed/LGG74_T1.nii.gz'),
        label=LabelMap('../BRATS2018_crop_renamed/LGG74_Label.nii.gz'),
    )

    subjects = [
        one_subject,
        another_subject,
    ]

    subjects_dataset = SubjectsDataset(subjects)
    queue_dataset = Queue(
        subjects_dataset,
        queue_length,
        patches_per_volume,
        UniformSampler(patch_size),
    )

    # This collate_fn is needed in the case of missing modalities
    # In this case, the batch will be composed by a *list* of samples instead
    # of the typical Python dictionary that is collated by default in Pytorch
    batch_loader = tio.SubjectsLoader(
        queue_dataset,
        batch_size=batch_size,
        collate_fn=lambda x: x,
    )

    # Mock PyTorch model
    model = nn.Identity()

    for epoch_index in range(num_epochs):
        logging.info('Epoch %s', epoch_index)
        for batch in batch_loader:  # batch is a *list* here, not a dictionary
            logits = model(batch)
            logging.info([batch[idx].keys() for idx in range(batch_size)])
            logging.info(logits.shape)
    logging.info('')


if __name__ == '__main__':
    main()
