# Use duecredit to provide a citation to relevant work to
# be cited. This does nothing unless the user has duecredit installed
# and calls this with duecredit (as in `python -m duecredit script.py`):
from .external.due import BibTeX
from .external.due import Doi
from .external.due import due

BIBTEX = r"""@article{perez-garcia_torchio_2021,
    title = {{TorchIO}: a {Python} library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning},
    journal = {Computer Methods and Programs in Biomedicine},
    pages = {106236},
    year = {2021},
    issn = {0169-2607},
    doi = {https://doi.org/10.1016/j.cmpb.2021.106236},
    url = {https://www.sciencedirect.com/science/article/pii/S0169260721003102},
    author = {P{\'e}rez-Garc{\'i}a, Fernando and Sparks, Rachel and Ourselin, S{\'e}bastien},
    keywords = {Medical image computing, Deep learning, Data augmentation, Preprocessing},
} """

TITLE = (
    'TorchIO: a Python library for efficient loading, preprocessing,'
    ' augmentation and patch-based sampling of medical images in deep learning'
)

DESCRIPTION = 'Tools for loading, augmenting and writing 3D medical images on PyTorch'

due.cite(
    BibTeX(BIBTEX),
    description=TITLE,
    path='torchio',
    cite_module=True,
)

due.cite(
    Doi('10.5281/zenodo.3739230'),
    description=DESCRIPTION,
    path='torchio',
    tags=['implementation'],
    cite_module=True,
)
