import numpy as np
from nbodykit.base.mesh import MeshFilter
from nbodykit.lab import FieldMesh


def compute_overdensity(
    eval_positions: np.array,
    tracers_mesh: FieldMesh,
    filter: MeshFilter,
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
    smooth_mesh = tracers_mesh.apply(
        filter,
        mode='complex',
        kind='wavenumber',
    )
    delta = smooth_mesh.paint(mode='real') - 1.
    return delta.readout(eval_positions)
