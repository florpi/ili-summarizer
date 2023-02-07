import numpy as np
from summarizer.data import Catalogue

def test_converting_to_mesh():
    n_tracers = 1000
    pos = np.random.random((n_tracers,3))
    boxsize = 1.
    cat = Catalogue(
        pos = pos,
        vel= None,
        redshift = 0.5,
        boxsize = boxsize,
        name = 'test',
        cosmo_dict = None
    )
    mesh = cat.to_mesh(n_mesh=10).preview()
    assert mesh.shape == (10,10,10)