import pandas as pd
import numpy as np
from pathlib import Path
import random
from .image import Image
from .subject import Subject
from .dataset import ImagesDataset

import torchio


class ImagesClassifDataset(ImagesDataset):
    def __init__(
            self,
            paths_dict,
            transform=None,
            verbose=False,
            infos=None,
            class_idx='noise',
            length=None,
            equal=None,
            ):
        """
        same as ImagesDataset but with a label for classification
        infos is a panda DataFrame
        length = to specify a reduce length if not all csv images
        """
        super().__init__(paths_dict, transform, verbose=verbose)

        lens = len(paths_dict)  #[len(paths) for paths in paths_dict.values()]
        if lens != infos.shape[0]:
            message = ('Suplementary collumns in csv file, should have the same number of lines as paths_dict')
            raise ValueError(message)

        self.infos = infos
        self.classes = infos[class_idx].values

        if length:
            self.length = length
        else:
            self.length = len(self.subjects)

        if equal:
            self.gen = self.get_data_equal
            print('sampling equal number in each classes')
        else:
            self.gen = self.get_data

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return next(self.gen(idx))

    def get_data_equal(self, idx):

        # np.random.seed( np.abs( int((time.time()-np.round(time.time()))*10000000) )) #if not numworker, gives the same data !
        # location = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/cache'
        # memory = Memory(location, verbose=0)
        # fonc_joblib = memory.cache(quadriview_adapt)

        indok = np.where(self.classes == 1)[0]
        indbad = np.where(self.classes == 0)[0]

        #w hile True:
        randtype = random.uniform(0, 1)  # np.random.rand()
        if randtype < 0.5:
            rand_idx = indbad[random.randint(0, len(indbad) - 1)]
        else:
            rand_idx = indok[random.randint(0, len(indok) - 1)]

        sample = super().__getitem__(rand_idx)
        one_info = self.infos.iloc[rand_idx, :].to_dict()
        sample.update(one_info)

        yield sample

    def get_data(self, index):
        sample = super().__getitem__(index)

        one_info = self.infos.iloc[index, :].to_dict()

        sample.update(one_info)
        # sample['infos'] = one_info

        yield sample


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


def get_subject_list_and_csv_info_from_data_prameters(data_param, fpath_idx="img_file", class_idx="noise",
                                          conditions=None, duplicate_class1=None, shuffle_order = True,
                                                      log = print):
    """
    :param data_param: same structure as for niftynet set script test/test_dataset.py
    :param fpath_idx: name of the collumn containing file path
    :param conditions : conditions to select lines. for instance column name corr conditions = [("corr", "<", 0.98),
            will select only values below 0.98
    :param duplicate_class1: number of time line from class_idx label 1 are duplicate. in this case it also randomly
            select same number of line from class 0 (to have equal class). No more usefull since you can get the exact
            list as it is in the csv file, and use equal=True in ImageClassifDataset
    :return: a paths_dict to be passed inot ImagesDataset and a panda dataframe containing all collumn to be passed
            as info in ImagesClassifDataset
    """
    paths_dict = dict()
    for key, vals in data_param.items():
        if 'csv_file' in vals:
            res = pd.read_csv(vals['csv_file'])

            log('Reading {} line in {}'.format(len(res), vals['csv_file']))
            log(' {} columns {}'.format(len(res.keys()), res.keys() ))

            if conditions is not None:
                res = res[apply_conditions_on_dataset(res, conditions)]
                nb_1, nb_0 = np.sum(res.loc[:, class_idx] == 1), np.sum(res.loc[:,class_idx] == 0)
                res.index = range(0, len(res))
                log('after condition %s / ok = %d / %d  = %f' % (class_idx, nb_1, nb_0, nb_0 / nb_1))

            if duplicate_class1 is not None:
                ii = np.argwhere(res.loc[:, class_idx] == 1)
                nbclass1 = len(ii)
                log('found {} of classe 1'.format(nbclass1))
                for kkk in range(0, duplicate_class1):
                    res = res.append(res.iloc[ii[:,0],:])
                nbclass1 = nbclass1 + nbclass1 * duplicate_class1
                log('after {} duplication found {} of classe 1'.format(duplicate_class1,nbclass1))

                ii1 =  np.argwhere(res.loc[:, class_idx] == 1)

                ii0 = np.argwhere(res.loc[:, class_idx] == 0)
                nbclass0 = len(ii0)
                select_class0 = np.random.randint(0, high=nbclass0, size=nbclass1)

                select_ind = np.vstack((ii1, ii0[select_class0])) #concatenate both class
                res = res.iloc[select_ind[:, 0], :]
                res.index = range(0, len(res))
                log('selecting same number of class 0 so we get a final size of  {}'.format(res.shape))

            if 'type' in vals :
                image_type = vals['type']
            else:
                image_type = torchio.INTENSITY

            # allfile = [Path(ff) for ff in res.loc[:, fpath_idx].str.strip()]
            if 'subjects_paths' in locals():
                new_subjects_path=[]
                for index, ff in enumerate(res.loc[:, fpath_idx].str.strip()):
                    paths_dict = subjects_paths[index]
                    dd = dict(path=ff, type=image_type)
                    paths_dict[key] = dd.copy()
                    new_subjects_path.append(paths_dict.copy())
                subjects_paths = new_subjects_path.copy()
            else:
                subjects_paths=[]
                for ff in res.loc[:, fpath_idx].str.strip():
                    dd = dict(path=ff, type=image_type)
                    paths_dict[key] = dd.copy()
                    subjects_paths.append(paths_dict.copy())

        else :
            log('key {} is not implemented (should be csv_file) '.fomat(vals.keys()))

    #shuffle the same way both subject_path_list and res
    if shuffle_order:
        from sklearn.utils import shuffle
        index = range(0, len(subjects_paths))
        index_shuffle = shuffle(index)
        res = res.reindex(index_shuffle)
        new_subjects_path = [subjects_paths[index_shuffle[ii]] for ii in index]
        subjects_paths = new_subjects_path

    #now convert the dictionary list into a list of Image
    subjects_list = []
    for subjects in subjects_paths:
        one_suj = {}
        for key, val in subjects.items():
            one_suj[key] = Image(val['path'], val['type'])
        new_suj = one_suj.copy()
        new_suj = Subject(new_suj)
        subjects_list.append(new_suj)

    return subjects_list, res

