from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import readfof


def load_sim(
    node: int,
    redshift: float,
    path_to_lhcs: Path,
) -> Tuple[np.array, np.array, np.array]:
    """load quijote simulationsn from latin hypercube

    Args:
        node (int): node to read
        redshift (float): redshift to read
        path_to_lhcs (_type_, optional): Path to where latin hypercubes are stored.

    Returns:
        pos (np.array): array of halo positions (in Mpc/h)
        vel (np.array): array of halo velocities (in km/s)
        mass (np.array): array of halo masses (in Msun/h)
    """
    snapshot_dict = {0.0: 4, 0.5: 3, 1.0: 2, 2.0: 1, 3.0: 0}
    FoF = readfof.FoF_catalog(
        # str(path_to_lhcs / f"{node}/"), #this is how Carolina had her version
        str(path_to_lhcs),
        snapshot_dict[redshift],
        long_ids=False,
        swap=False,
        SFR=False,
        read_IDs=False,
    )
    pos = FoF.GroupPos / 1e3
    vel = FoF.GroupVel * (1.0 + redshift)
    mass = FoF.GroupMass * 1e10
    return pos, vel, mass

# Carolina had this coded in but it is apparnetly not necessary
# def load_params_sim(
#     node: int,
#     path_to_lhcs: Path,
# ) -> Dict[str, float]:
#     """Read the cosmological parameters of the node
#
#     Args:
#         node (int): node to read
#         path_to_lhcs (Optional[Path], optional): path to latin hypercubes.
#
#     Returns:
#         Dict[str, float]: dictionary of cosmological parametesr
#     """
#
#     return (
#         pd.read_csv(path_to_lhcs/ "latin_hypercube_params.txt", delimiter=r"\s+")
#         .iloc[node]
#         .to_dict()
#     )
