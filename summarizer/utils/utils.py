import numpy as np
from nbodykit.base.mesh import MeshFilter
from nbodykit.lab import FieldMesh

import logging
import datetime
import os
from os.path import join as pjoin


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


def get_source_path(cfg, simtype, check=True):
    wdir = cfg.meta.wdir
    nbody = cfg.nbody

    # get the path to the source directory, and check at each level
    sim_dir = pjoin(wdir, nbody.suite, simtype)
    cfg_dir = pjoin(sim_dir, f'L{nbody.L}-N{nbody.N}')
    lh_dir = pjoin(cfg_dir, str(nbody.lhid))

    if check:
        if not os.path.isdir(sim_dir):
            raise ValueError(
                f"Simulation directory {sim_dir} does not exist.")
        if not os.path.isdir(cfg_dir):
            raise ValueError(
                f"Configuration directory {cfg_dir} does not exist.")
        if not os.path.isdir(lh_dir):
            raise ValueError(
                f"Latin hypercube directory {lh_dir} does not exist.")
    return lh_dir


def timing_decorator(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        logging.info(f"Running {func.__name__}...")
        t0 = datetime.datetime.now()
        out = func(*args, **kwargs)
        dt = (datetime.datetime.now() - t0).total_seconds()
        logging.info(
            f"Finished {func.__name__}... "
            f"({int(dt//60)}m{int(dt%60)}s)")
        return out
    return wrapper


@timing_decorator
def load_params(index, cosmofile):
    if index == "fid":
        return [0.3175, 0.049, 0.6711, 0.9624, 0.834]
    with open(cosmofile, 'r') as f:
        content = f.readlines()[index]
    content = [float(x) for x in content.split()]
    return content