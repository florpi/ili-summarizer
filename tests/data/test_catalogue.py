import numpy as np
from summarizer.data import BoxCatalogue

def test_converting_to_mesh():
    n_tracers = 1000
    pos = np.random.random((n_tracers,3))
    boxsize = 1.
    cat = BoxCatalogue(
        galaxies_pos = pos,
        redshift = 0.5,
        boxsize = boxsize,
        name = 'test',
    )
    mesh = cat.get_mesh(n_mesh=10)
    assert mesh.shape == (10,10,10)