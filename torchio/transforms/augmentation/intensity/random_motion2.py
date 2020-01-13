import math
import torch
import numpy as np
from scipy.interpolate import pchip_interpolate
try:
    import finufftpy
    finufft = True
except ImportError:
    finufft = False

from torchio import  INTENSITY
from .. import RandomTransform


def create_rotation_matrix_3d(angles):
    """
    given a list of 3 angles, create a 3x3 rotation matrix that describes rotation about the origin
    :param angles (list or numpy array) : rotation angles in 3 dimensions
    :return (numpy array) : rotation matrix 3x3
    """

    mat1 = np.array([[1., 0., 0.],
                     [0., math.cos(angles[0]), math.sin(angles[0])],
                     [0., -math.sin(angles[0]), math.cos(angles[0])]],
                    dtype='float')

    mat2 = np.array([[math.cos(angles[1]), 0., -math.sin(angles[1])],
                     [0., 1., 0.],
                     [math.sin(angles[1]), 0., math.cos(angles[1])]],
                    dtype='float')

    mat3 = np.array([[math.cos(angles[2]), math.sin(angles[2]), 0.],
                     [-math.sin(angles[2]), math.cos(angles[2]), 0.],
                     [0., 0., 1.]],
                    dtype='float')

    mat = (mat1 @ mat2) @ mat3
    return mat


class MotionSimTransform(RandomTransform):

    def __init__(self, std_rotation_angle=0, std_translation=10,
                 corrupt_pct=(15, 20), freq_encoding_dim=(0, 1, 2), preserve_center_pct=0.07,
                 apply_mask=True, nufft=False, proc_scale=-1, num_pieces=8, verbose=False):
        """
        :param image_name (str): key in data dictionary
        :param std_rotation_angle (float) : std of rotations
        :param std_translation (float) : std of translations
        :param corrupt_pct (list of ints): range of percents
        :param freq_encoding_dim (list of ints): randomly choose freq encoding dim
        :param preserve_center_pct (float): percentage of k-space center to preserve
        :param apply_mask (bool): apply mask to output or not
        :param nufft (bool): whether to use nufft for introducing rotations
        :param proc_scale (float or int) : -1 = piecewise, -2 = uncorrelated, 0 = retroMocoBox or float for random walk scale
        :param num_pieces (int): number of pieces for piecewise constant simulation
       raises ImportError if nufft is true but finufft cannot be imported
        """

        super(MotionSimTransform, self).__init__(verbose=verbose)
        self.trajectory = None
        self.preserve_center_frequency_pct = preserve_center_pct
        self.freq_encoding_choice = freq_encoding_dim
        self.frequency_encoding_dim = None
        self.proc_scale = proc_scale
        self.num_pieces = num_pieces
        self.std_rotation_angle, self.std_translation = std_rotation_angle, std_translation
        self.corrupt_pct_range = corrupt_pct
        self.apply_mask = apply_mask

        if self.proc_scale == -1:
            self._simulate_trajectory = self._piecewise_simulation
            print('using piecewise motion simulation')
        elif self.proc_scale == -2:
            self._simulate_trajectory = self._gaussian_simulation
            print('using uncorrelated gaussian simulation')
        elif self.proc_scale == 0:
            self._simulate_trajectory = self._simul_motion
            print('using RetroMocoBox algorithm')
        elif self.proc_scale > 0:
            self._simulate_trajectory = self._random_walk_simulation
            print('using random walk')
        else:
            raise ValueError('invalid proc_scale: should be either 0, -1,-2 or positive real valued')

        self.nufft = nufft
        if (not finufft) and nufft:
            raise ImportError('finufftpy cannot be imported')

        self.frequency_encoding_dim = np.random.choice(self.freq_encoding_choice)

    def apply_transform(self, sample):
        ###############################
        ########## T E S T ############
        ###############################

        for image_name, image_dict in sample.items():
            if not isinstance(image_dict, dict) or 'type' not in image_dict:
                # Not an image
                continue
            if image_dict['type'] != INTENSITY:
                continue
            image_data = np.squeeze(image_dict['data'])[..., np.newaxis, np.newaxis]
            original_image = np.squeeze(image_data[:, :, :, 0, 0])
            self._calc_dimensions(original_image.shape)
            self._simulate_random_trajectory()

            # fft
            im_freq_domain = self._fft_im(original_image)
            translated_im_freq_domain = self._translate_freq_domain(freq_domain=im_freq_domain)

            # iNufft for rotations
            if self.nufft:
                corrupted_im = self._nufft(translated_im_freq_domain)
                corrupted_im = corrupted_im / corrupted_im.size  # normalize

            else:
                corrupted_im = self._ifft_im(translated_im_freq_domain)

            # magnitude
            corrupted_im = abs(corrupted_im)
            image_dict["data"] = corrupted_im[np.newaxis, ...]
            image_dict['data'] = torch.from_numpy(image_dict['data'])

            """
            if self.apply_mask:
                # todo: use input arg mask
                mask_im = input_data['mask'][:, :, :, 0, 0] > 0
                corrupted_im = np.multiply(corrupted_im, mask_im)
                masked_original = np.multiply(original_image, mask_im)
                image_data[:, :, :, 0, 0] = masked_original

            #image_data[:, :, :, 0, 1] = corrupted_im

        #output_data = input_data
        #output_data[self.image_name] = image_data
        """
        return sample

    @staticmethod
    def get_params():
        pass

    def _calc_dimensions(self, im_shape):
        """
        calculate dimensions based on im_shape
        :param im_shape (list/tuple) : image shape
        - sets self.phase_encoding_dims, self.phase_encoding_shape, self.num_phase_encoding_steps, self.frequency_encoding_dim
        - initializes self.translations and self.rotations
        """
        pe_dims = [0, 1, 2]
        pe_dims.pop(self.frequency_encoding_dim)
        self.phase_encoding_dims = pe_dims
        im_shape = list(im_shape)
        self.im_shape = im_shape.copy()
        im_shape.pop(self.frequency_encoding_dim)
        self.phase_encoding_shape = im_shape
        self.num_phase_encoding_steps = self.phase_encoding_shape[0] * self.phase_encoding_shape[1]
        self.translations = np.zeros(shape=(3, self.num_phase_encoding_steps))
        self.rotations = np.zeros(shape=(3, self.num_phase_encoding_steps))
        self.frequency_encoding_dim = len(self.im_shape) - 1 if self.frequency_encoding_dim == -1 \
            else self.frequency_encoding_dim

    @staticmethod
    def random_walk_trajectory(length, start_scale=10, proc_scale=0.1):
        seq = np.zeros([3, length])
        seq[:, 0] = np.random.normal(loc=0.0, scale=start_scale, size=(3,))
        for i in range(length - 1):
            seq[:, i + 1] = seq[:, i] + np.random.normal(scale=proc_scale, size=(3,))
        return seq

    @staticmethod
    def piecewise_trajectory(length, n_pieces=4, scale_trans=10, scale_rot=3):
        """
        generate random piecewise constant trajectory with n_pieces
        :param length (int): length of trajectory
        :param n_pieces (int): number of pieces
        :param scale_trans (float): scale of normal distribution for translations
        :param scale_rot (float): scale of normal distribution for rotations
        :return: list of numpy arrays of size (3 x length) for translations and rotations
        """
        seq_trans = np.zeros([3, length])
        seq_rot = np.zeros([3, length])
        ind_to_split = np.random.choice(length, size=n_pieces)
        split_trans = np.array_split(seq_trans, ind_to_split, axis=1)
        split_rot = np.array_split(seq_rot, ind_to_split, axis=1)
        for i, sp in enumerate(split_trans):
            sp[:] = np.random.normal(scale=scale_trans, size=(3, 1))
        for i, sp in enumerate(split_rot):
            sp[:] = np.random.normal(scale=scale_rot, size=(3, 1))
        return seq_trans, seq_rot

    def _random_walk_simulation(self, length):
        rand_translations = self.random_walk_trajectory(length,
                                                        start_scale=self.std_translation,
                                                        proc_scale=self.proc_scale)
        rand_rotations = self.random_walk_trajectory(length,
                                                     start_scale=self.std_rotation_angle,
                                                     proc_scale=self.std_rotation_angle / 1000)
        return rand_translations, rand_rotations

    def _piecewise_simulation(self, length):
        num_pieces = np.random.choice(np.arange(1, self.num_pieces))
        rand_translations, rand_rotations = self.piecewise_trajectory(length, n_pieces=num_pieces,
                                                                      scale_trans=self.std_translation,
                                                                      scale_rot=self.std_rotation_angle)
        return rand_translations, rand_rotations

    def _gaussian_simulation(self, length):
        rand_translations = np.random.normal(size=[3, length], scale=self.std_translation)
        rand_rotations = np.random.normal(size=[3, length], scale=self.std_rotation_angle)
        return rand_translations, rand_rotations

    def _center_k_indices_to_preserve(self):
        """get center k indices of freq domain"""
        mid_pts = [int(math.ceil(x / 2)) for x in self.phase_encoding_shape]
        num_pts_preserve = [math.ceil(self.preserve_center_frequency_pct * x) for x in self.phase_encoding_shape]
        ind_to_remove = {val + 1: slice(mid_pts[i] - num_pts_preserve[i], mid_pts[i] + num_pts_preserve[i])
                         for i, val in enumerate(self.phase_encoding_dims)}
        ix_to_remove = [ind_to_remove.get(dim, slice(None)) for dim in range(4)]
        return ix_to_remove

    @staticmethod
    def perlinNoise1D(npts, weights):
        if not isinstance(weights, list):
            weights = range(int(round(weights)))
            weights = np.power([2] * len(weights), weights)

        n = len(weights)
        xvals = np.linspace(0, 1, npts)
        total = np.zeros((npts, 1))

        for i in range(n):
            frequency = 2 ** (i)
            this_npts = round(npts / frequency)

            if this_npts > 1:
                total += weights[i] * pchip_interpolate(np.linspace(0, 1, this_npts), np.random.random((this_npts, 1)),
                                                        xvals)
            else:

                print("Maxed out at octave {}".format(i))

        total = total - np.min(total)
        total = total / np.max(total)
        return total.reshape(-1)

    def _simul_motion(self, motion_lines):
        """
        Exemple:
        noiseBasePars = 6.5
        swallowFrequency = 2.1243
        swallowMagnitude = [4.23554] * 2
        suddenFrequency = 4.3434
        suddenMagnitude = [4.24424] * 2
        displacement_shift = 2
        """
        noiseBasePars, swallowFrequency, swallowMagnitude, suddenFrequency, suddenMagnitude = np.random.uniform(0.5,
                                                                                                                5.0,
                                                                                                                size=5)
        displacement_shift = np.random.randint(0, 5, size=1)
        swallowMagnitude = [swallowMagnitude]*2
        suddenMagnitude = [suddenMagnitude]*2
        nT = self.im_shape[self.frequency_encoding_dim]
        maxRot, maxDisp = self.std_rotation_angle, self.std_translation
        fitpars = np.zeros((6, nT))

        if noiseBasePars > 0:
            fitpars[0, :] = maxDisp * (self.perlinNoise1D(nT, noiseBasePars) - 0.5)
            fitpars[1, :] = maxDisp * (self.perlinNoise1D(nT, noiseBasePars) - 0.5)
            fitpars[2, :] = maxDisp * (self.perlinNoise1D(nT, noiseBasePars) - 0.5)

            fitpars[3, :] = maxRot * (self.perlinNoise1D(nT, noiseBasePars) - 0.5)
            fitpars[4, :] = maxRot * (self.perlinNoise1D(nT, noiseBasePars) - 0.5)
            fitpars[5, :] = maxRot * (self.perlinNoise1D(nT, noiseBasePars) - 0.5)
        # add in swallowing-like movements - just to z direction and pitch
        if swallowFrequency > 0:
            swallowTraceBase = np.exp(-np.linspace(0, 100, nT))
            swallowTrace = np.zeros((nT))

            for i in range(int(round(swallowFrequency))):
                rand_shifts = int(round(np.random.rand() * nT))
                rolled = np.roll(swallowTraceBase, rand_shifts, axis=0)
                swallowTrace += rolled

            fitpars[2, :] += swallowMagnitude[0] * swallowTrace
            fitpars[3, :] += swallowMagnitude[1] * swallowTrace

        # add in random sudden movements in any direction
        if suddenFrequency > 0:
            suddenTrace = np.zeros(fitpars.shape)

            for i in range(int(round(suddenFrequency))):
                iT_sudden = int(np.ceil(np.random.rand() * nT))
                to_add = np.asarray([suddenMagnitude[0] * (2 * np.random.random(3) - 1),
                                     suddenMagnitude[1] * (2 * np.random.random(3) - 1)]).reshape((-1, 1))
                suddenTrace[:, iT_sudden:] = np.add(suddenTrace[:, iT_sudden:], to_add)

            fitpars += suddenTrace

        if displacement_shift > 0:
            to_substract = fitpars[:, int(round(nT / 2))]
            fitpars = np.subtract(fitpars, to_substract[..., np.newaxis])
        """
        displacements = np.sqrt(np.sum(fitpars[:3] ** 2, axis=0))
        rotations = np.sqrt(np.sum(fitpars[3:] ** 2, axis=0))

        dict_params = {
            'displacements': displacements,
            'RMS_displacements': np.sqrt(np.mean(displacements ** 2)),
            'rotations': rotations,
            'RMS_rot': np.sqrt(np.mean(rotations ** 2))
        }
        """
        print(f' in _simul_motionfitpar shape fitpars {fitpars.shape}')

        to_reshape = [1, 1, 1]
        phase_dime = self.im_shape[self.frequency_encoding_dim]

        x1 = np.linspace(0,1,phase_dime)
        x2 = np.linspace(0,1,np.prod(self.im_shape))
        fitpars_interp=[]
        for ind in range(fitpars.shape[0]):
            y = fitpars[ind, :]
            yinterp = np.interp(x2,x1,y)
            fitpars_interp.append(yinterp)

        fitpars = np.array(fitpars_interp)

        rand_translations, rand_rotations = fitpars[:3].reshape([3] + self.im_shape), fitpars[3:].reshape([3] + self.im_shape)

        #to_reshape[self.frequency_encoding_dim] = -1
        #rand_translations, rand_rotations = fitpars[:3].reshape([3] + to_reshape), fitpars[3:].reshape([3] + to_reshape)

        #ones_multiplicator = np.ones([3] + self.im_shape)
        #rand_translations, rand_rotations = rand_translations * ones_multiplicator, rand_rotations * ones_multiplicator

        print(f' in _simul_motionfitpar shape rand_rotation {rand_rotations.shape}')

        return rand_translations, rand_rotations

    def _simulate_random_trajectory(self):
        """
        simulates random trajectory using a random number of lines generated from corrupt_pct_range
        modifies self.translations and self.rotations
        """

        # Each voxel has a random translation and rotation for 3 dimensions.
        rand_translations_vox = np.zeros([3] + self.im_shape)
        rand_rotations_vox = np.zeros([3] + self.im_shape)

        # randomly choose PE lines to corrupt
        choose_from_list = [np.arange(i) for i in self.phase_encoding_shape]
        num_lines = [int(x / 100 * np.prod(self.phase_encoding_shape)) for x in self.corrupt_pct_range]

        # handle deterministic case where no range is given
        if num_lines[0] == num_lines[1]:
            num_lines = num_lines[0]
        else:
            num_lines = np.random.randint(num_lines[0], num_lines[1], size=1)

        if num_lines == 0:
            # allow no lines to be modified
            self.translations = rand_translations_vox.reshape(3, -1)
            self.rotations = rand_rotations_vox.reshape(3, -1)
            return

        motion_lines = []
        for i in range(len(self.phase_encoding_shape)):
            motion_lines.append(np.random.choice(choose_from_list[i], size=num_lines, replace=True).tolist())

        # sort by either first or second PE dim
        dim_to_sort_by = np.random.choice([0, 1])
        motion_lines_sorted = [list(x) for x in
                               zip(*sorted(zip(motion_lines[0], motion_lines[1]), key=lambda x: x[dim_to_sort_by]))]
        motion_lines = motion_lines_sorted

        # generate random motion parameters
        rand_translations, rand_rotations = self._simulate_trajectory(len(motion_lines[0]))

        if self.proc_scale == 0:
            #self.translations, self.rotations = rand_translations, rand_rotations
            rand_translations_vox, rand_rotations_vox = rand_translations, rand_rotations

        else:
            # create indexing tuple ix
            motion_ind_dict = {self.phase_encoding_dims[i]: val for i, val in enumerate(motion_lines)}
            ix = [motion_ind_dict.get(dim, slice(None)) for dim in range(3)]
            ix = tuple(ix)

            # expand in freq-encoding dim
            new_dims = [3, rand_translations.shape[-1]]
            self.rand_translations = np.expand_dims(rand_translations, -1)
            self.rand_rotations = np.expand_dims(rand_rotations, -1)
            new_dims.append(self.im_shape[self.frequency_encoding_dim])
            self.rand_translations = np.broadcast_to(self.rand_translations, new_dims)
            self.rand_rotations = np.broadcast_to(self.rand_rotations, new_dims)

            # insert into voxel-wise motion parameters
            for i in range(3):
                rand_rotations_vox[(i,) + ix] = self.rand_rotations[i, :, :]
                rand_translations_vox[(i,) + ix] = self.rand_translations[i, :, :]

        ix_to_remove = self._center_k_indices_to_preserve()
        rand_translations_vox[tuple(ix_to_remove)] = 0
        rand_rotations_vox[tuple(ix_to_remove)] = 0

        self.translations = rand_translations_vox.reshape(3, -1)
        rand_rotations_vox = rand_rotations_vox.reshape(3, -1)
        self.rotations = rand_rotations_vox * (math.pi / 180.)  # convert to radians

        np.save('/tmp/rp_mot.npy',np.vstack([ self.translations, self.rotations]) )

    def gen_test_trajectory(self, translation, rotation):
        """
        # for testing - apply the same transformation at all Fourier (time) points
        :param translation (list/array of length 3):
        :param rotation (list/array of length 3):
        modifies self.translations, self.rotations in place
        """
        num_pts = np.prod(self.im_shape)
        self.translations = np.array([np.ones([num_pts, ]).flatten() * translation[0],
                                      np.ones([num_pts, ]).flatten() * translation[1],
                                      np.ones([num_pts, ]).flatten() * translation[2]])

        self.rotations = np.array([np.ones([num_pts, ]).flatten() * rotation[0],
                                   np.ones([num_pts, ]).flatten() * rotation[1],
                                   np.ones([num_pts, ]).flatten() * rotation[2]])

    def _fft_im(self, image):
        output = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))).astype(np.complex128)
        return output

    def _ifft_im(self, freq_domain):
        output = np.fft.ifftshift(np.fft.ifftn(freq_domain))
        return output

    def _translate_freq_domain(self, freq_domain):
        """
        image domain translation by adding phase shifts in frequency domain
        :param freq_domain - frequency domain data 3d numpy array:
        :return frequency domain array with phase shifts added according to self.translations:
        """

        lin_spaces = [np.linspace(-0.5, 0.5, x) for x in freq_domain.shape]
        meshgrids = np.meshgrid(*lin_spaces, indexing='ij')
        grid_coords = np.array([mg.flatten() for mg in meshgrids])

        phase_shift = np.multiply(grid_coords, self.translations).sum(axis=0)  # phase shift is added
        exp_phase_shift = np.exp(-2j * math.pi * phase_shift)
        freq_domain_translated = np.multiply(exp_phase_shift, freq_domain.flatten(order='C')).reshape(freq_domain.shape)

        return freq_domain_translated

    def _rotate_coordinates(self):
        """
        :return: grid_coordinates after applying self.rotations
        """
        center = [math.ceil((x - 1) / 2) for x in self.im_shape]

        [i1, i2, i3] = np.meshgrid(np.arange(self.im_shape[0]) - center[0],
                                   np.arange(self.im_shape[1]) - center[1],
                                   np.arange(self.im_shape[2]) - center[2], indexing='ij')

        grid_coordinates = np.array([i1.T.flatten(), i2.T.flatten(), i3.T.flatten()])

        print('rotation size is {}'.format( self.rotations.shape))

        rotations = self.rotations.reshape([3] + self.im_shape)
        ix = (len(self.im_shape) + 1) * [slice(None)]
        ix[self.frequency_encoding_dim + 1] = 0  # dont need to rotate along freq encoding

        rotations = rotations[tuple(ix)].reshape(3, -1)
        rotation_matrices = np.apply_along_axis(create_rotation_matrix_3d, axis=0, arr=rotations).transpose([-1, 0, 1])
        rotation_matrices = rotation_matrices.reshape(self.phase_encoding_shape + [3, 3])
        rotation_matrices = np.expand_dims(rotation_matrices, self.frequency_encoding_dim)

        rotation_matrices = np.tile(rotation_matrices,
                                    reps=([self.im_shape[
                                               self.frequency_encoding_dim] if i == self.frequency_encoding_dim else 1
                                           for i in range(5)]))  # tile in freq encoding dimension

        rotation_matrices = rotation_matrices.reshape([-1, 3, 3])

        # tile grid coordinates for vectorizing computation
        grid_coordinates_tiled = np.tile(grid_coordinates, [3, 1])
        grid_coordinates_tiled = grid_coordinates_tiled.reshape([3, -1], order='F').T
        rotation_matrices = rotation_matrices.reshape([-1, 3])

        print('rotation matrices size is {}'.format(rotation_matrices.shape))

        new_grid_coords = (rotation_matrices * grid_coordinates_tiled).sum(axis=1)

        # reshape new grid coords back to 3 x nvoxels
        new_grid_coords = new_grid_coords.reshape([3, -1], order='F')

        # scale data between -pi and pi
        max_vals = [abs(x) for x in grid_coordinates[:, 0]]
        new_grid_coordinates_scaled = [(new_grid_coords[i, :] / max_vals[i]) * math.pi for i in
                                       range(new_grid_coords.shape[0])]
        new_grid_coordinates_scaled = [np.asfortranarray(i) for i in new_grid_coordinates_scaled]

        return new_grid_coordinates_scaled, [grid_coordinates, new_grid_coords]

    def _nufft(self, freq_domain_data, iflag=1, eps=1E-7):
        """
        rotate coordinates and perform nufft
        :param freq_domain_data:
        :param iflag/eps: see finufftpy doc
        :param eps: precision of nufft
        :return: nufft of freq_domain_data after applying self.rotations
        """

        if not finufft:
            raise ImportError('finufftpy not available')

        new_grid_coords = self._rotate_coordinates()[0]

        # initialize array for nufft output
        f = np.zeros([len(new_grid_coords[0])], dtype=np.complex128, order='F')

        freq_domain_data_flat = np.asfortranarray(freq_domain_data.flatten(order='F'))

        finufftpy.nufft3d1(new_grid_coords[0], new_grid_coords[1], new_grid_coords[2], freq_domain_data_flat,
                           iflag, eps, self.im_shape[0], self.im_shape[1],
                           self.im_shape[2], f, debug=0, spread_debug=0, spread_sort=2, fftw=0, modeord=0,
                           chkbnds=0, upsampfac=1.25)  # upsampling at 1.25 saves time at low precisions
        im_out = f.reshape(self.im_shape, order='F')

        return im_out


"""
################################################################## T E S T #############################################
import nibabel as nb
import numpy as np
import os
import sys
from os.path import join as opj
sys.path.extend("/home/ghiles.reguig/Work/torchio/torchio/")
from torchio.dataset import ImagesDataset
from torchvision.transforms import Compose
from torch.utils.data import DataLoader


def read_hcp_data(fpath):
    list_data = []
    for dir_suj in os.listdir(fpath):
        if dir_suj.startswith("suj_"):
            path_suj = opj(fpath, dir_suj, "spm12", "T1w_acpc_dc_restore.nii")
            if os.path.exists(path_suj):
                dict_suj = {"T1w": dict(path=path_suj, type="intensity", suj=dir_suj)}
                list_data.append(dict_suj)
    return list_data


def save_data(path, data_dict, suj):
    data = data_dict["data"][0]
    affine = data_dict["affine"]
    nb.Nifti1Image(data, affine).to_filename(opj(path, suj))


fpath = "/data/romain/HCPdata"
list_paths = read_hcp_data(fpath)
l_paths = list_paths[:10]

t = MotionSimTransform(std_rotation_angle=5, std_translation=10, nufft=True, proc_scale=0)
transform = Compose([t])

ds_moco = ImagesDataset(l_paths, transform=transform)

dataloader_moco = DataLoader(ds_moco, batch_size=1, collate_fn=lambda x: x)

dst_save = "/data/ghiles/motion_simulation/retromocobox/"

for idx, data in enumerate(dataloader_moco):
    print("Processing {}".format(idx))
    d = data[0]["T1w"]
    save_data(dst_save, d, "suj_{}".format(idx))

########################################################### T E S T S with txt
import pandas as pd
import nibabel as nb
import numpy as np

# Read mvt params (6 x 219)
data = pd.read_csv("/data/romain/HCPdata/suj_100307/Motion/rp_Motion_RMS_185_Disp_325_swalF_0_swalM_0_sudF_0_sudM_0_Motion_RMS_0_Disp_0_swalF_T1w_1mm.txt", header=None)
data = data.drop(columns=[218])
#Extract separately parameters of translation and rotation
trans_data, rot_data = data[:3].values.reshape((3, 1, 218, -1)), data[3:].values.reshape((3, 1, 218, -1))
#Read MRI data
mri_data = nb.load("/data/romain/HCPdata/suj_100307/T1w_1mm.nii.gz")
image = mri_data.get_data()
# Squeezed img
squeezed_img = np.squeeze(image)

fitpars=data.values
x1 = np.linspace(0,1,data.shape[1])
x2 = np.linspace(0,1,np.prod(image.shape[1:])) #interp on the phase_encoding * slice_endocing
fitpars_interp=[]
for ind in range(fitpars.shape[0]):
    y = fitpars[ind, :]
    yinterp = np.interp(x2,x1,y)
    fitpars_interp.append(yinterp)

fitpars = np.array(fitpars_interp)

translations, rotations = fitpars[:3], fitpars[3:]
rotations = np.radians(rotations)

trans_data, rot_data = fitpars[:3].reshape((3, 1, 218, -1)), fitpars[3:].reshape((3, 1, 218, -1))
ones_multiplicator = np.ones([3] + list(squeezed_img.shape))
translations, rotations = trans_data * ones_multiplicator, rot_data * ones_multiplicator
translations, rotations = translations.reshape((3, -1)), rotations.reshape((3, -1))
rotations = np.radians(rotations)

# BAD replication
#Broadcast translation and rotation parameters to voxel level params
ones_multiplicator = np.ones([3] + list(squeezed_img.shape))
translations, rotations = trans_data * ones_multiplicator, rot_data * ones_multiplicator
##To voxel-wise params
translations, rotations = translations.reshape((3, -1)), rotations.reshape((3, -1))
rotations = np.radians(rotations)

#### TRANSFORMATION

t = MotionSimTransform(std_rotation_angle=5, std_translation=10, nufft=True)
t.frequency_encoding_dim = 1 #On fixe la dimension d'encodage de fréquence
t._calc_dimensions(squeezed_img.shape)
# Set rotations and translation params
t.translations, t.rotations = translations, rotations
im_freq_domain = t._fft_im(squeezed_img)

#Translation
translated_freq = t._translate_freq_domain(im_freq_domain)
translated_img = t._ifft_im(translated_freq)
corr_trans = abs(translated_img)

#Rotation
rotated_freq = t._nufft(translated_freq)
corr_rot = abs(rotated_freq) / rotated_freq.size

#Save data
corr_trans_img = nb.Nifti1Image(corr_trans, mri_data.affine)
#corr_trans_img.to_filename("/data/ghiles/motion_simulation/tests/translated.nii")

corr_rot_img = nb.Nifti1Image(corr_rot, mri_data.affine)
corr_rot_img.to_filename("/tmp/rotated2.nii")

nT = 218
nt_pf = np.round(nT/6)
fitpars = np.zeros((6, nT))
noiseBasePars = 6.5
maxDisp = 2.181541
maxRot = 2.181541
swallowFrequency = 2.1243
swallowMagnitude = [4.23554] * 2
suddenFrequency = 4.3434
suddenMagnitude = [4.24424] * 2
displacement_shift = 2

disp_params, desc_params = simul_motion(nT=nT, noiseBasePars=noiseBasePars, maxDisp=maxDisp, maxRot=maxRot,
                   swallowFrequency=swallowFrequency, swallowMagnitude=swallowMagnitude, suddenFrequency=suddenFrequency,
                 suddenMagnitude=suddenMagnitude, displacement_shift=displacement_shift)
"""
