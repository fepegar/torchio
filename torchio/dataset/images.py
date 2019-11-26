import numpy as np
import nibabel as nib
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(self, paths_dict, transform=None, add_bg_to_label=False):
        """
        paths_dict is expected to have keys: image, [,label[, sampler]]
        TODO: pixel size, orientation (for now assume RAS 1mm iso)
        Caveat: all images must have the same shape so that they can be
        collated by a DataLoader. TODO: write custom collate_fn?
        """
        self.paths_dict = paths_dict
        self.transform = transform
        self.add_bg_to_label = add_bg_to_label

    def __len__(self):
        return len(self.paths_dict['image'])

    def __getitem__(self, index):
        # Load image
        image, affine, image_path = self.load_image('image', index)
        sample = dict(
            image=image,
            path=image_path,
            affine=affine,
        )

        # Load label
        if 'label' in self.paths_dict:
            label = self.load_data('label', index)
            if self.add_bg_to_label:
                label = self.add_background(label)
            sample['label'] = label

        # Load sampler
        if 'sampler' in self.paths_dict:
            sampler = self.load_data('sampler', index)
            sample['sampler'] = sampler

        # Apply transform (bottleneck)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def transform_and_save(sample, output_paths_dict, extract_fg=False):
        for key, output_path in output_paths_dict.items():
            data = sample[key].squeeze()
            if key == 'label' and extract_fg:
                data = data[1]
            affine = sample['affine']
            nii = nib.Nifti1Image(data, affine)
            nii.header['qform_code'] = 1
            nii.header['sform_code'] = 0
            nii.to_filename(str(output_path))

    def load_image(self, key, index):
        """
        https://github.com/nipy/dmriprep/issues/55#issuecomment-448322366
        """
        path = self.paths_dict[key][index]
        img = nib.load(path)
        data = np.array(img.dataobj)
        data = data[np.newaxis, ...]  # add channels dimension
        return data, img.affine, path

    @staticmethod
    def add_background(label):
        """
        Adds a background channel to label array
        """
        foreground = label
        background = 1 - label
        label = np.concatenate((background, foreground))
        return label

    def load_data(self, key, index):
        path = self.paths_dict[key][index]
        img = nib.load(path)
        data = np.array(img.dataobj)
        data = data[np.newaxis, ...]  # add channels dimension
        return data
