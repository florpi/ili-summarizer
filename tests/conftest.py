import pytest
import numpy as np
from summarizer.data import Catalogue


@pytest.fixture(name="dummy_catalogue", autouse=True)
def make_dummy_catalogue():
    n_tracers = 100
    pos = np.random.random((n_tracers, 3))
    vel = np.random.random((n_tracers, 3))
    mass = np.random.random((n_tracers,))
    redshift = 0.5
    boxsize = 1.0
    cosmo_dict = {"dummy": 1.0}
    name = "dummy_n0"
    return Catalogue(
        pos=pos,
        vel=vel,
        redshift=redshift,
        boxsize=boxsize,
        cosmo_dict=cosmo_dict,
        name=name,
        mass=mass,
    )
