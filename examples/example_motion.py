import nibabel as nib
from torchio import ImagesDataset, transforms, INTENSITY

paths = [{'t1': dict(path='~/Dropbox/MRI/t1.nii.gz', type=INTENSITY)}]
dataset = ImagesDataset(paths)
sample = dataset[0]
transform = transforms.RandomMotion(
    seed=42,
    degrees=10,
    translation=10,
    num_transforms=2,
)
transformed = transform(sample)
nib.Nifti1Image(
    transformed['t1']['data'].squeeze(),
    transformed['t1']['affine'],
).to_filename('/tmp/move.nii.gz')
