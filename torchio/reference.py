# Use duecredit to provide a citation to relevant work to
# be cited. This does nothing unless the user has duecredit installed
# and calls this with duecredit (as in `python -m duecredit script.py`):
from .external.due import due, Doi, BibTeX

BIBTEX = r"""@misc{fern2020torchio,
   title={TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning},
   author={Fernando Pérez-García and Rachel Sparks and Sebastien Ourselin},
   year={2020},
   eprint={2003.04696},
   archivePrefix={arXiv},
   primaryClass={eess.IV}
} """  # noqa: E501

TITLE = (
    'TorchIO: a Python library for efficient loading, preprocessing,'
    ' augmentation and patch-based sampling of medical images in deep learning'
)

DESCRIPTION = (
    'Tools for loading, augmenting and writing 3D medical images'
    ' on PyTorch'
)

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
