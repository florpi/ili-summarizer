import numpy as np
from pypower import CatalogMesh

class Gaussian(object):
    '''Gaussian filter in Fourier space

    Parameters
    ----------
    r : float
        the radius of the Gaussian filter
    '''
    def __init__(self, r):
        self.r = r

    def __call__(self, k, v):
        r = self.r
        k2 = sum(ki ** 2 for ki in k)
        return np.exp(- 0.5 * k2 * r**2) * v

def compute_overdensity(
    query_positions: np.array,
    catalogue,
    smoothing_radius: float,
    compensate=False,
)->np.array:
    """ Evaluate overdensity at positions ```eval_positions```, given a mesh
    with tracers and a filter to smooth the mesh.

    Args:
        eval_positions (np.array): positions in which to evaluate the overdensity. 
        tracers_mesh (Mesh): mesh with tracers. 
        filter (MeshFilter): filter to smooth the mesh. 

    Returns:
        np.array: overdensity values at ```eval_positions```. 
    """
    data_mesh = catalogue.galaxies_mesh
    gaussian_filter = Gaussian(r=smoothing_radius)
    data_mesh = data_mesh.r2c().apply(gaussian_filter,)
    data_mesh = data_mesh.c2r()
    if catalogue.has_randoms:
        randoms_mesh = CatalogMesh(
            data_positions=catalogue.galaxies_pos,
            data_weights=catalogue.galaxies_weights,
            nmesh=catalogue.n_mesh,
            resampler=catalogue.mesh_resampler,
            position_type='pos',
            boxsize=catalogue.boxsize,
            boxcenter=[catalogue.boxsize / 2.0, catalogue.boxsize / 2.0, catalogue.boxsize / 2.0],
            wrap=False,
        )
        randoms_mesh = randoms_mesh.to_mesh(field='data-normalized_randoms',compensate=compensate)
        randoms_mesh = randoms_mesh.r2c().apply(gaussian_filter,)
        randoms_mesh = randoms_mesh.c2r()
        sum_data, sum_randoms = np.sum(data_mesh.value), np.sum(randoms_mesh.value)
        alpha = sum_data / sum_randoms
        delta_mesh = data_mesh - alpha * randoms_mesh
        mask = randoms_mesh > 0
        delta_mesh[mask] /= alpha * randoms_mesh[mask]
        delta_mesh[~mask] = 0.0
    else:
        sum_data = np.sum(data_mesh)
        delta_mesh = data_mesh/np.mean(data_mesh) - 1
    return delta_mesh.readout(query_positions, resampler=catalogue.mesh_resampler)
