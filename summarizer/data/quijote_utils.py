from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import readfof


def load_sim(
    node: int,
    redshift: float,
    path_to_lhcs=Path("/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/quijote/"),
    massive_neutrinos: bool = False,
) -> Tuple[np.array, np.array, np.array]:
    """load quijote simulationsn from latin hypercube

    Args:
        node (int): node to read
        redshift (float): redshift to read
        path_to_lhcs (_type_, optional): Path to where latin hypercubes are stored. Defaults to Path("/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/quijote/").
        massive_neutrinos (bool, optional): wheter to use massive neutrinos. Defaults to False.

    Returns:
        pos (np.array): array of halo positions (in Mpc/h)
        vel (np.array): array of halo velocities (in km/s)
        mass (np.array): array of halo masses (in Msun/h)
    """
    if massive_neutrinos:
        data_dir = path_to_lhcs / f"latin_hypercube_nwLH/"
    else:
        data_dir = path_to_lhcs / f"latin_hypercube/"
    snapshot_dict = {0.0: 4, 0.5: 3, 1.0: 2, 2.0: 1, 3.0: 0}
    FoF = readfof.FoF_catalog(
        str(data_dir / f"{node}/"),
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


def load_params_sim(
    node: int,
    path_to_lhcs: Optional[Path] = Path(
        "/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/quijote/"
    ),
    massive_neutrinos: bool = False,
) -> Dict[str, float]:
    """Read the cosmological parameters of the node

    Args:
        node (int): node to read
        path_to_lhcs (Optional[Path], optional): path to latin hypercubes. Defaults to Path( "/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/quijote/" ).
        massive_neutrinos (bool, optional): whether to use massive neutrino simulations. Defaults to False.

    Returns:
        Dict[str, float]: dictionary of cosmological parametesr
    """

    if massive_neutrinos:
        data_dir = path_to_lhcs / f"latin_hypercube_nwLH/"
    else:
        data_dir = path_to_lhcs / f"latin_hypercube/"
    return (
        pd.read_csv(data_dir / "latin_hypercube_params.txt", delimiter=r"\s+")
        .iloc[node]
        .to_dict()
    )
