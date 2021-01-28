from .fpg import FPG
from .slicer import Slicer
from .episurg import EPISURG
from .ixi import IXI, IXITiny
from .itk_snap import BrainTumor, T1T2, AorticValve
from .mni import Colin27, Sheep, Pediatric, ICBM2009CNonlinearSymmetric


__all__ = [
    'FPG',
    'Slicer',
    'IXI',
    'IXITiny',
    'EPISURG',
    'BrainTumor',
    'T1T2',
    'AorticValve',
    'Colin27',
    'Sheep',
    'Pediatric',
    'ICBM2009CNonlinearSymmetric',
]
