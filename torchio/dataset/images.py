from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path



class ImagesDataset(Dataset):
    def __init__(self, paths_dict, transform=None, add_bg_to_label=False):
        """
        paths_dict is expected to have keys: image, [,label[, sampler]]
        TODO: pixel size, orientation (for now assume RAS 1mm iso)
        Caveat: all images must have the same shape so that they can be
        collated by a DataLoader. TODO: write custom collate_fn?
        """
        paths_dict = paths_dict.copy()
        self.sujid = paths_dict.pop('sujid', None) #this will remove field sujid if exist
        print(paths_dict.keys())

        self.parse_paths_dict(paths_dict)
        self.paths_dict = paths_dict
        self.transform = transform
        self.add_bg_to_label = add_bg_to_label

    def __len__(self):
        return len(self.paths_dict['image'])

    def __getitem__(self, index):
        sample = {}
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else -1

        for key in self.paths_dict:
            if key == 'image':
                image, affine, image_path = self.load_image(key, index)
                image_dict = dict(
                    image=image,
                    path=str(image_path),
                    affine=affine,
                )
                sample.update(image_dict)
            elif key == 'label':
                label = self.load_data('label', index)
                if self.add_bg_to_label:
                    label = self.add_background(label)
                sample['label'] = label
            else:
                data = self.load_data(key, index)
                sample[key] = data

        # Apply transform (bottleneck)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.sujid is not None:
            print('Woker {} get index {} sujid {}'.format(worker_id,index, self.sujid[index]))
        else : print('Woker {} get index {} '.format(worker_id,index))

        return sample

    def load_image(self, key, index, add_channels_dim=True):
        """
        https://github.com/nipy/dmriprep/issues/55#issuecomment-448322366
        """
        path = self.paths_dict[key][index]
        img = nib.load(str(path))
        data = np.array(img.dataobj)
        if add_channels_dim:
            data = data[np.newaxis, ...]  # add channels dimension
        return data, img.affine, path

    def load_data(self, key, index, add_channels_dim=True):
        path = self.paths_dict[key][index]
        img = nib.load(str(path))
        data = np.array(img.dataobj)
        if add_channels_dim:
            data = data[np.newaxis, ...]  # add channels dimension

        return data

    @staticmethod
    def parse_paths_dict(paths_dict):
        lens = [len(paths) for paths in paths_dict.values()]
        if sum(lens) == 0:
            raise ValueError('All paths lists are empty')
        if len(set(lens)) > 1:
            message = (
                'Paths lists have different lengths:'
            )
            for key, paths in paths_dict.items():
                message += f'\n{key}: {len(paths)}'
            raise ValueError(message)
        for paths_list in paths_dict.values():
            for path in paths_list:
                path = Path(path)
                if not path.is_file():
                    raise FileNotFoundError(f'{path} not found')

    @staticmethod
    def add_background(label):
        """
        Adds a background channel to label array
        """
        foreground = label
        background = 1 - label
        label = np.concatenate((background, foreground))
        return label

    @staticmethod
    def save_sample(sample, output_paths_dict, extract_fg=False):
        for key, output_path in output_paths_dict.items():
            data = sample[key].squeeze()
            if key == 'label' and extract_fg:
                data = data[1]
            affine = sample['affine']
            nii = nib.Nifti1Image(data, affine)
            nii.header['qform_code'] = 1
            nii.header['sform_code'] = 0
            nii.to_filename(str(output_path))


def get_paths_dict_from_data_prameters(data_param):
    """
    :param data_param: same structure as for niftynet set script test/test_dataset.py
    :return:
    """
    paths_dict = dict()
    for key, vals in data_param.items():
        if 'csv_file' in vals:
            csvfile = pd.read_csv(vals['csv_file'], header=None)

            print('Reading {} line in {}'.format(len(csvfile), vals['csv_file']))
            allfile = [Path(ff) for ff in  csvfile.loc[:, 1].str.strip()]

            sujid   = csvfile.loc[:, 0].values
            paths_dict[key] = allfile

            if 'sujid' in paths_dict :
                #test if same subject id
                if np.array_equal(sujid,paths_dict['sujid'] ) is False:
                    message =("First column subject ID differs")
                    raise ValueError(message)
            else :
                paths_dict['sujid'] = sujid

        else :
            print('key {} is not implemented (should be csv_file) '.fomat(vals.keys()))

    return paths_dict
