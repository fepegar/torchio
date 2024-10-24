import urllib.parse

from ..constants import DATA_REPO
from ..data import LabelMap
from ..data import ScalarImage
from ..data.io import read_matrix
from ..data.subject import Subject
from ..download import download_url
from ..utils import get_torchio_cache_dir


class FPG(Subject):
    """3T :math:`T_1`-weighted brain MRI and corresponding parcellation.

    Args:
        load_all: If ``True``, three more images will be loaded: a
            :math:`T_2`-weighted MRI, a diffusion MRI and a functional MRI.
    """

    def __init__(self, load_all: bool = False):
        repo_dir = urllib.parse.urljoin(DATA_REPO, 'fernando/')

        self.filenames = {
            't1': 't1.nii.gz',
            'seg': 't1_seg_gif.nii.gz',
            'rigid': 't1_to_mni.tfm',
            'affine': 't1_to_mni_affine.h5',
        }
        if load_all:
            self.filenames['t2'] = 't2.nii.gz'
            self.filenames['fmri'] = 'fmri.nrrd'
            self.filenames['dmri'] = 'dmri.nrrd'

        download_root = get_torchio_cache_dir() / 'fpg'

        for filename in self.filenames.values():
            download_url(
                urllib.parse.urljoin(repo_dir, filename),
                download_root,
                filename=filename,
            )

        rigid = read_matrix(download_root / self.filenames['rigid'])
        affine = read_matrix(download_root / self.filenames['affine'])
        subject_dict = {
            't1': ScalarImage(
                download_root / self.filenames['t1'],
                rigid_matrix=rigid,
                affine_matrix=affine,
            ),
            'seg': LabelMap(
                download_root / self.filenames['seg'],
                rigid_matrix=rigid,
                affine_matrix=affine,
            ),
        }
        if load_all:
            subject_dict['t2'] = ScalarImage(
                download_root / self.filenames['t2'],
            )
            subject_dict['fmri'] = ScalarImage(
                download_root / self.filenames['fmri'],
            )
            subject_dict['dmri'] = ScalarImage(
                download_root / self.filenames['dmri'],
            )
        super().__init__(subject_dict)
        self.gif_colors = self.GIF_COLORS

    def plot(self, *args, **kwargs):
        super().plot(*args, **kwargs, cmap_dict={'seg': self.gif_colors})

    GIF_COLORS = {
        0: (0, 0, 0),
        1: (0, 0, 0),
        5: (127, 255, 212),
        12: (240, 230, 140),
        16: (176, 48, 96),
        24: (48, 176, 96),
        31: (48, 176, 96),
        32: (103, 255, 255),
        33: (103, 255, 255),
        35: (238, 186, 243),
        36: (119, 159, 176),
        37: (122, 186, 220),
        38: (122, 186, 220),
        39: (96, 204, 96),
        40: (96, 204, 96),
        41: (220, 247, 164),
        42: (220, 247, 164),
        43: (205, 62, 78),
        44: (205, 62, 78),
        45: (225, 225, 225),
        46: (225, 225, 225),
        47: (60, 60, 60),
        48: (220, 216, 20),
        49: (220, 216, 20),
        50: (196, 58, 250),
        51: (196, 58, 250),
        52: (120, 18, 134),
        53: (120, 18, 134),
        54: (255, 165, 0),
        55: (255, 165, 0),
        56: (12, 48, 255),
        57: (12, 48, 225),
        58: (236, 13, 176),
        59: (236, 13, 176),
        60: (0, 118, 14),
        61: (0, 118, 14),
        62: (165, 42, 42),
        63: (165, 42, 42),
        64: (160, 32, 240),
        65: (160, 32, 240),
        66: (56, 192, 255),
        67: (56, 192, 255),
        70: (255, 225, 225),
        72: (184, 237, 194),
        73: (180, 231, 250),
        74: (225, 183, 231),
        76: (180, 180, 180),
        77: (180, 180, 180),
        81: (245, 255, 200),
        82: (255, 230, 255),
        83: (245, 245, 245),
        84: (220, 255, 220),
        85: (220, 220, 220),
        86: (200, 255, 255),
        87: (250, 220, 200),
        89: (245, 255, 200),
        90: (255, 230, 255),
        91: (245, 245, 245),
        92: (220, 255, 220),
        93: (220, 220, 220),
        94: (200, 255, 255),
        96: (140, 125, 255),
        97: (140, 125, 255),
        101: (255, 62, 150),
        102: (255, 62, 150),
        103: (160, 82, 45),
        104: (160, 82, 45),
        105: (165, 42, 42),
        106: (165, 42, 42),
        107: (205, 91, 69),
        108: (205, 91, 69),
        109: (100, 149, 237),
        110: (100, 149, 237),
        113: (135, 206, 235),
        114: (135, 206, 235),
        115: (250, 128, 114),
        116: (250, 128, 114),
        117: (255, 255, 0),
        118: (255, 255, 0),
        119: (221, 160, 221),
        120: (221, 160, 221),
        121: (0, 238, 0),
        122: (0, 238, 0),
        123: (205, 92, 92),
        124: (205, 92, 92),
        125: (176, 48, 96),
        126: (176, 48, 96),
        129: (152, 251, 152),
        130: (152, 251, 152),
        133: (50, 205, 50),
        134: (50, 205, 50),
        135: (0, 100, 0),
        136: (0, 100, 0),
        137: (173, 216, 230),
        138: (173, 216, 230),
        139: (153, 50, 204),
        140: (153, 50, 204),
        141: (160, 32, 240),
        142: (160, 32, 240),
        143: (0, 206, 208),
        144: (0, 206, 208),
        145: (51, 50, 135),
        146: (51, 50, 135),
        147: (135, 50, 74),
        148: (135, 50, 74),
        149: (218, 112, 214),
        150: (218, 112, 214),
        151: (240, 230, 140),
        152: (240, 230, 140),
        153: (255, 255, 0),
        154: (255, 255, 0),
        155: (255, 110, 180),
        156: (255, 110, 180),
        157: (0, 255, 255),
        158: (0, 255, 255),
        161: (100, 50, 100),
        162: (100, 50, 100),
        163: (178, 34, 34),
        164: (178, 34, 34),
        165: (255, 0, 255),
        166: (255, 0, 255),
        167: (39, 64, 139),
        168: (39, 64, 139),
        169: (255, 99, 71),
        170: (255, 99, 71),
        171: (255, 69, 0),
        172: (255, 69, 0),
        173: (210, 180, 140),
        174: (210, 180, 140),
        175: (0, 255, 127),
        176: (0, 255, 127),
        177: (74, 155, 60),
        178: (74, 155, 60),
        179: (255, 215, 0),
        180: (255, 215, 0),
        181: (238, 0, 0),
        182: (238, 0, 0),
        183: (46, 139, 87),
        184: (46, 139, 87),
        185: (238, 201, 0),
        186: (238, 201, 0),
        187: (102, 205, 170),
        188: (102, 205, 170),
        191: (255, 218, 185),
        192: (255, 218, 185),
        193: (238, 130, 238),
        194: (238, 130, 238),
        195: (255, 165, 0),
        196: (255, 165, 0),
        197: (255, 192, 203),
        198: (255, 192, 203),
        199: (244, 222, 179),
        200: (244, 222, 179),
        201: (208, 32, 144),
        202: (208, 32, 144),
        203: (34, 139, 34),
        204: (34, 139, 34),
        205: (125, 255, 212),
        206: (127, 255, 212),
        207: (0, 0, 128),
        208: (0, 0, 128),
    }


# For backward compatibility
GIF_COLORS = FPG.GIF_COLORS
