import warnings
from pathlib import Path
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(
            self,
            paths_dict,
            transform=None,
            verbose=False,
            ):
        """
        paths_dict is expected to have keys: image, [,label[, sampler[, *]]]
        TODO: pixel size, orientation (for now assume RAS 1mm iso)
        Caveat: if using whole image for training,
        all images must have the same shape so that they can be
        collated by a DataLoader. TODO: write custom collate_fn?
        """
        self.parse_paths_dict(paths_dict)
        self.paths_dict = paths_dict
        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.paths_dict['image'])

    def __getitem__(self, index):
        sample = {}
        for key in self.paths_dict:
            if key == 'image':
                image, affine, image_path = self.load_image(key, index)
                image_dict = dict(
                    image=image,
                    path=str(image_path),
                    affine=affine,
                )
                sample.update(image_dict)
            else:
                data = self.load_data(key, index)
                sample[key] = data

        # Apply transform (bottleneck)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def load_image(self, key, index, add_channels_dim=True):
        """
        https://github.com/nipy/dmriprep/issues/55#issuecomment-448322366
        """
        path = self.paths_dict[key][index]
        if self.verbose:
            print(f'Loading {path}...')
        img = nib.load(str(path))
        data = np.array(img.dataobj).astype(np.float32)
        if self.verbose:
            print(f'Loaded array with shape {data.shape}')
        if data.ndim != 3:
            message = (
                f'Image {path} has shape {data.shape}.'
                ' Keeping only [..., 0]'
            )
            warnings.warn(message)
        if add_channels_dim:
            data = data[np.newaxis, ...]  # add channels dimension
        data = data[..., 0] if data.ndim == 5 else data
        return data, img.affine, path

    def load_data(self, key, index):
        path = self.paths_dict[key][index]
        img = nib.load(str(path))
        data = np.array(img.dataobj).astype(np.float32)
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
