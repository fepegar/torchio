from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from ..utils import get_stem

class ImagesDataset(Dataset):
    def __init__(
            self,
            paths_dict,
            transform=None,
            verbose=False,
            ):
        """
        paths_dict is expected to have keys: image, [,label[, sampler[, *]]]
        For example:
        paths_dict = dict(
            image=images_paths,
            label=labels_paths,
        )
        If using whole image for training, all images must have the
        same shape so that they can be collated by a DataLoader.
        TODO: write custom collate_fn?
        TODO: handle pixel size, orientation (for now assume RAS 1mm iso)
        """
        paths_dict = paths_dict.copy()
        self.sujid = paths_dict.pop('sujid', None) #this will remove field sujid if exist
        print(paths_dict.keys())

        self.parse_paths_dict(paths_dict)
        self.paths_dict = paths_dict
        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.paths_dict['image'])

    def __getitem__(self, index):
        sample = {}
        #worker = torch.utils.data.get_worker_info()
        #worker_id = worker.id if worker is not None else -1

        for key in self.paths_dict:
            data, affine, image_path = self.load_image(key, index)
            sample[key] = data
            if key == 'image':
                image_dict = dict(
                    path=str(image_path),
                    affine=affine,
                    stem=get_stem(image_path),
                    sujid=self.sujid[index] if self.sujid is not None else 'TODO',
                )
                sample.update(image_dict)

        label_name = [kk for kk in sample if ('label' in kk) ]
        if len(label_name) > 1:
            list_label = [sample[kkk].squeeze() for kkk in label_name]
            for kkk in label_name :
                del sample[kkk]  #remove label_1 label_2 ... entery
            sample['label'] = np.stack(list_label)

        # Apply transform (this is usually the major bottleneck)
        if self.transform is not None:
            sample = self.transform(sample)

        #if self.sujid is not None:
        #    print('Woker {} get index {} sujid {}'.format(worker_id,index, self.sujid[index]))
        #else : print('Woker {} get index {} '.format(worker_id,index))

        return sample

    def load_image(self, key, index, add_channels_dim=True):
        path = self.paths_dict[key][index]
        if self.verbose:
            print(f'Loading {path}...')
        img = nib.load(str(path))

        # See https://github.com/nipy/dmriprep/issues/55#issuecomment-448322366
        data = np.array(img.dataobj).astype(np.float32)

        if self.verbose:
            print(f'Loaded array with shape {data.shape}')
        num_dimensions = data.ndim
        if num_dimensions > 3:
            message = (
                f'Processing of {num_dimensions}D volumes not supported.'
                f' {path} has shape {data.shape}'
            )
            raise NotImplementedError(message)
        data = data[np.newaxis, ...] if add_channels_dim else data
        return data, img.affine, path

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
    def save_sample(sample, output_paths_dict):
        for key, output_path in output_paths_dict.items():
            data = sample[key].squeeze()
            affine = sample['affine']
            nii = nib.Nifti1Image(data, affine)
            nii.header['qform_code'] = 1
            nii.header['sform_code'] = 0
            nii.to_filename(str(output_path))


class ImagesClassifDataset(ImagesDataset):
    def __init__(
            self,
            paths_dict,
            transform=None,
            verbose=False,
            infos=None,
            ):
        """
        same as ImagesDataset but with a label for classification
        infos is a panda DataFrame
        """
        super().__init__(paths_dict, transform, verbose=verbose)

        lens = [len(paths) for paths in paths_dict.values()]
        if lens[0] != infos.shape[0]:
            message =('Suplementary collumns in csv file, should have the same number of lines as paths_dict')
            raise ValueError(message)

        self.infos = infos

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        one_info = self.infos.iloc[index, :].to_dict()

        sample.update(one_info)
        #sample['infos'] = one_info

        return sample


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
            allfile = [Path(ff) for ff in csvfile.loc[:, 1].str.strip()]

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

def apply_conditions_on_dataset(dataset, conditions, min_index=None, max_index=None):
    """
    Conditions of the form ((intermediate_op,) var, op, values).
    Takes one or several conditions (each condition must be in a 3-tuple or a 4-tuple, each block of conditions in a list)
    and a dataframe, and returns the part of the dataframe respecting the conditions.
    """
    import operator
    operator_dict = {
        '==': operator.eq,
        '>': operator.gt,
        '<': operator.lt,
        "|": operator.or_,
        "&": operator.and_
    }

    if max_index is not None or min_index is not None:
        if max_index is None:
            dataset = dataset.iloc[min_index:]
        elif min_index is None:
            dataset = dataset.iloc[:max_index]
        else:
            dataset = dataset.iloc[min_index:max_index]
    for p_c in conditions:
        if type(p_c) == list:
            if len(p_c[0]) == 3:
                snap = apply_conditions_on_dataset(dataset, p_c)
            else:
                temp_var = p_c[0][0]
                p_c[0] = p_c[0][1:]
                temp_snap = apply_conditions_on_dataset(dataset, p_c)
                snap = operator_dict[temp_var](snap.copy(), temp_snap)
        elif len(p_c) == 3:
            snap = operator_dict[p_c[1]](dataset.copy()[p_c[0]], p_c[2])
        elif len(p_c) == 4:
            snap = operator_dict[p_c[0]]((snap.copy()), (operator_dict[p_c[2]](dataset.copy()[p_c[1]], p_c[3])))
    return snap


def get_paths_and_res_from_data_prameters(data_param, fpath_idx="img_file",
                                          conditions=None,):
    """
    :param data_param: same structure as for niftynet set script test/test_dataset.py
    :param fpath_idx: name of the collumn containing file path
    :return: a paths_dict to be passed inot ImagesDataset and a panda dataframe containing all collumn to be passed
            as info in ImagesClassifDataset
    """
    paths_dict = dict()
    for key, vals in data_param.items():
        if 'csv_file' in vals:
            res = pd.read_csv(vals['csv_file'])

            print('Reading {} line in {}'.format(len(res), vals['csv_file']))
            print(' {} columns {}'.format(len(res.keys()), res.keys() ))

            if conditions is not None:
                res = res[apply_conditions_on_dataset(res, conditions)]
                nb_1, nb_0 = np.sum(res.noise == 1), np.sum(res.noise == 0)
                print('after condition noise/ok = %d / %d  = %f' % (nb_1, nb_0, nb_0 / nb_1))

            allfile = [Path(ff) for ff in res.loc[:, fpath_idx].str.strip()]

            paths_dict[key] = allfile

        else :
            print('key {} is not implemented (should be csv_file) '.fomat(vals.keys()))

    return paths_dict, res
