# Try importing backends. If they work, import their dependent module.
# Current backends: nbodykit, pycorr, jax, kymatio
from .two_point import TwoPCF, Pk, Mk
from .environment import DensitySplit
notfound = []
try:
    import jax, jaxlib
    from .knn import KNN
except ModuleNotFoundError:
    notfound.append('jax')

try:
    import Corrfunc
    from .vpf import VPF
    from .cic import CiC
except ModuleNotFoundError:
    notfound.append('Corrfunc')

try:
    import kymatio
    import torch
    from .wavelet import WST
except ModuleNotFoundError:
    notfound.append('kymatio or torch')

try:
    import PolyBin3D
    from .three_point import Bk
except ModuleNotFoundError:
    notfound.append('PolyBin3D')



if len(notfound)>0:
    import warnings
    warnings.warn(f"Running without the following backends due to ModuleNotFoundError: {', '.join(notfound)}")
