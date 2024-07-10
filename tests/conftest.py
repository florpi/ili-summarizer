import pytest
import numpy as np
from summarizer.data import BoxCatalogue


@pytest.fixture(name="dummy_catalogue", autouse=True)
def make_dummy_catalogue():
    n_tracers = 100
    pos = np.random.random((n_tracers, 3))
    redshift = 0.5
    boxsize = 1.0
    name = "dummy_n0"
    return BoxCatalogue(
        galaxies_pos=pos,
        redshift=redshift,
        boxsize=boxsize,
        name=name,
    )
