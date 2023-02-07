import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Union
import nbodykit.lab as nblab 

class Catalogue:
    def __init__(
        self,
        pos: np.array,
        vel: np.array,
        redshift: float,
        boxsize: float,
        cosmo_dict: Dict[str, float],
        name: str,
        mesh: bool = True,
        n_mesh: Optional[int] = 360,
    ):
        """Catalogue of tracers (dark matter halos, galaxies...)

        Args:
            pos (np.array): 3D vector of positions, size (N, 3) 
            vel (np.array): 3D vector of velocities, size (N, 3) 
            redshift (float): redshift of the catalogue
            boxsize (float): size of the simulation box
            cosmo_dict (Dict[str, float]): dictionary fo cosmological parameters
            mesh (bool, optional): whether to create a mesh. Defaults to True.
            n_mesh (Optional[int], optional): number of cells in the mesh. Defaults to 50.
        """
        self.pos = pos % boxsize  
        self.vel = vel
        self.redshift = redshift
        self.boxsize = boxsize
        self.cosmo_dict = cosmo_dict
        self.name = name
        if mesh:
            self.mesh = self.to_mesh(n_mesh=n_mesh)

    def __str__(self,)->str:
        """get name for catalogue

        Returns:
            str: name
        """
        return self.name

    def __len__(
        self,
    ) -> int:
        """ Get number of objects in catalogue

        Returns:
            int: n tracers
        """
        return len(self.pos)

    @classmethod
    def from_quijote(
        cls,
        node: int,
        redshift: float,
        path_to_lhcs: Union[Path,str], 
        n_halos: Optional[int] = None,
        n_density_halos: Optional[float] = None,
        los: Optional[int] = 2,
    ) -> "Catalogue":
        """Get a catalogue for the quijote simulations latin hyper cube

        Args:
            node (int): node to read
            redshift (float): redshift, one of 0.0, 0.5, 1.0, 2.0, 3.0
            n_halos (Optional[int], optional): Number of halos to include. Defaults to None.
            n_density_halos (Optional[int], optional): Number density of halos to select. Defaults to None.
            path_to_lhcs (Path, optional): Path to latin hypercube data. 

        Returns:
            Catalogue: catalogue for simulation
        """
        import redshift_space_library as RSL
        from summarizer.data.quijote_utils import load_params_sim, load_sim

        path_to_lhcs = Path(path_to_lhcs)
        pos, vel, mass = load_sim(
            node=node, redshift=redshift, path_to_lhcs=path_to_lhcs
        )
        boxsize = 1000.0
        if n_halos is None and n_density_halos is not None:
            n_halos = int(n_density_halos * boxsize **3)
        if n_halos is not None:
            sorted_mass_idx = np.argsort(mass)
            pos = pos[sorted_mass_idx][-n_halos:, :]
            vel = vel[sorted_mass_idx, :][-n_halos:, :]
        cosmo_dict = load_params_sim(node=node, path_to_lhcs=path_to_lhcs)
        if los is not None:
            Omega_l = 1.0 - cosmo_dict["Omega_m"]
            Hubble = 100.0 * np.sqrt(
                cosmo_dict["Omega_m"] * (1.0 + redshift) ** 3 + Omega_l
            )
            RSL.pos_redshift_space(pos, vel, boxsize, Hubble, redshift, axis=los)
        return cls(
            pos=pos,
            vel=vel,
            redshift=redshift,
            cosmo_dict=cosmo_dict,
            boxsize=boxsize,
            name=f'quijote_node{node}'
        )
    
    def to_nbodykit_catalogue(self,)->nblab.ArrayCatalog:
        """ Get a nbodykit catalogue from the catalogue

        Returns:
            nblab.ArrayCatalog: nbodykit catalogue 
        """
        return nblab.ArrayCatalog(
                {'Position': self.pos}, 
                BoxSize=self.boxsize, 
                dtype=np.float32, 
        ) 

    def to_mesh(self, n_mesh: int, resampler: str = "tsc") -> np.array:
        """Get a mesh from the catalogue

        Args:
            n_mesh (int): number of cells in the mesh
            resampler (str, optional): resampler to use. Defaults to "tsc".

        Returns:
            np.array: mesh
        """
        nblab_cat = self.to_nbodykit_catalogue()
        return nblab_cat.to_mesh(
            Nmesh=n_mesh, 
            resampler=resampler
        )

