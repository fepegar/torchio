from pprint import pprint
import nibabel as nib
from torchio import ImagesDataset, transforms, INTENSITY

paths = [{
    't1': dict(path='~/Dropbox/MRI/t1.nii.gz', type=INTENSITY),
    'colin': dict(path='/tmp/colin27_t1_tal_lin.nii.gz', type=INTENSITY),
}]

paths = [{
    't1': dict(path='/data/romain/data_exemple/mni/MNI152_T1_1mm.nii.gz', type=INTENSITY),
    'colin': dict(path='/data/romain/data_exemple/mni/mean_nr1000/Mean_S50_all.nii', type=INTENSITY),
}]

dataset = ImagesDataset(paths)
sample = dataset[0]
transform = transforms.RandomMotion(
    seed=42,
    degrees=20,
    translation=15,
    num_transforms=3,
    verbose=True,
)
transformed = transform(sample)

pprint(transformed['t1']['random_motion_times'])

nib.Nifti1Image(
    transformed['t1']['data'].squeeze(),
    transformed['t1']['affine'],
).to_filename('/tmp/t1_motion.nii.gz')

nib.Nifti1Image(
    transformed['colin']['data'].squeeze(),
    transformed['colin']['affine'],
).to_filename('/tmp/colin_motion.nii.gz')
