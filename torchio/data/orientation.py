import warnings
import nibabel as nib


AXCODES_TO_WORDS = {
    'L': 'left',
    'R': 'right',
    'P': 'posterior',
    'A': 'anterior',
    'I': 'inferior',
    'S': 'superior',
    # 'C': 'caudal',
    # 'R': 'rostral',  # conflic with right
    # 'D': 'dorsal',
    # 'V': 'ventral',
}


def name_dimensions(tensor, affine):
    axcodes = nib.aff2axcodes(affine)
    names = [AXCODES_TO_WORDS[axcode] for axcode in axcodes]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        tensor.rename_(*names)
