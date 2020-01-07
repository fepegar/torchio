from pprint import pprint
from torchio import ImagesDataset, transforms, INTENSITY

paths = [{
    't1': dict(path='~/Dropbox/MRI/t1.nii.gz', type=INTENSITY),
    'colin': dict(path='/tmp/colin27_t1_tal_lin.nii.gz', type=INTENSITY),
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

dataset.save_sample(transformed, dict(t1='/tmp/t1_motion.nii.gz'))
dataset.save_sample(transformed, dict(colin='/tmp/colin_motion.nii.gz'))
