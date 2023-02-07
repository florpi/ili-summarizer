import numpy as np
import pytest
from pathlib import Path
from summarizer.utils import compute_overdensity
from summarizer.data import Catalogue
from nbodykit.filters import TopHat


def test__compute_overdensity():
    cat = Catalogue.from_quijote(
        node=0, 
        redshift=0.5,
        path_to_lhcs= Path('/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/quijote/latin_hypercube'),
    )
    top_hat = TopHat(r=15.)
    delta = compute_overdensity(cat.pos, cat.mesh, filter=top_hat)
    assert len(delta) == len(cat.pos)
    assert np.mean(delta) > 0.
    assert np.mean(delta) < 100.


