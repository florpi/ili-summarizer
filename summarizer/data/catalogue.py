import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Union
from abc import abstractmethod

try:
    import nbodykit.lab as nblab
except ModuleNotFoundError:
    pass


class BaseCatalogue:
    def __init__(
        self,
        galaxies_pos: np.array,
        redshift: float,
        weights: Optional[np.array] = None,
        boxsize: Optional[float] = None,
        name: Optional[str] = None,
        n_mesh: Optional[int] = 360,
    ):
        """Base catalogue object for manipulating pointclouds

        Args:
            galaxies_pos (np.array): 3D positions x,y,z, of shape (N_tracers, 3)
            redshift (float): redshift of the catalogue
            weights (Optional[np.array], optional): weights for the tracers. Defaults to None.
            boxsize (Optional[float], optional): simulation box size, None if not periodic cubic box. Defaults to None.
            name (Optional[str], optional): catalogue name. Defaults to None.
            n_mesh (Optional[int], optional): number of cells in mesh. Defaults to 360.
        """

        self.galaxies_pos = galaxies_pos
        self.weights = weights
        self.boxsize = boxsize
        self.name = name
        self.redshift = redshift
        self.n_mesh = n_mesh
        if self.n_mesh is not None:
            self.mesh = self.to_mesh(
                n_mesh=n_mesh,
                weights=self.weights,
            )



    def __str__(
        self,
    ) -> str:
        """get name for catalogue

        Returns:
            str: name
        """
        return self.name

    def __len__(
        self,
    ) -> int:
        """Get number of objects in catalogue

        Returns:
            int: n tracers
        """
        return len(self.galaxies_pos)

    @abstractmethod
    def to_mesh(
        self,
    ):
        return


    @property
    def is_periodic_box(self,):
        return True if self.boxsize is not None else False

class BoxCatalogue(BaseCatalogue):
    def __init__(
        self,
        galaxies_pos: np.array,
        boxsize: float,
        redshift: float,
        cosmology: Dict[str, float],
        weights: Optional[np.array] = None,
        name: Optional[str] = None,
        n_mesh: Optional[int] = 360,
    ):
        """ Catalogue for a simulated box

        Args:
            galaxies_pos (np.array): galaxy positions x,y,z
            boxsize (float):  simulation box size
            redshift (float): redshift of the catalogue 
            weights (Optional[np.array], optional): weights for the tracers. Defaults to None.
            cosmology (Dict[str, float]): true cosmology of the simultion 
            name (Optional[str], optional): catalogue name. Defaults to None.
            n_mesh (Optional[int], optional): number of cells in mesh. Defaults to 360.
        """
        super().__init__(
            self,
            galaxies_pos=galaxies_pos,
            redshift=redshift,
            name=name,
            weights=weights,
            n_mesh=n_mesh,
            boxsize=boxsize,
        )
        self.cosmology = cosmology

    @classmethod
    def from_quijote(
        cls,
        node: int,
        redshift: float,
        path_to_lhcs: Union[Path, str],
        n_halos: Optional[int] = None,
        n_density_halos: Optional[float] = None,
        los: Optional[int] = 2,
        n_mesh: Optional[int] = 360,
        boxsize: Optional[float] = 1000.0,
    ) -> "Catalogue":
        """Get a catalogue for the quijote simulations latin hyper cube

        Args:
            node (int): node to read
            redshift (float): redshift, one of 0.0, 0.5, 1.0, 2.0, 3.0
            n_halos (Optional[int], optional): Number of halos to include. Defaults to None.
            n_density_halos (Optional[int], optional): Number density of halos to select. Defaults to None.
            path_to_lhcs (Path, optional): Path to latin hypercube data.
            mesh (bool, optional): whether to create a mesh. Defaults to True.
            n_mesh (Optional[int], optional): number of cells in the mesh. Defaults to 50.

        Returns:
            Catalogue: catalogue for simulation
        """
        from summarizer.data.quijote_utils import load_params_sim, load_sim

        path_to_lhcs = Path(path_to_lhcs)
        pos, vel, mass = load_sim(
            node=node, redshift=redshift, path_to_lhcs=path_to_lhcs
        )
        if n_halos is None and n_density_halos is not None:
            n_halos = int(n_density_halos * boxsize**3)
        if n_halos is not None:
            sorted_mass_idx = np.argsort(mass)
            pos = pos[sorted_mass_idx][-n_halos:, :]
            mass = mass[sorted_mass_idx][-n_halos:]
            vel = vel[sorted_mass_idx, :][-n_halos:, :]
        cosmo_dict = load_params_sim(node=node, path_to_lhcs=path_to_lhcs)
        if los is not None:
            Omega_l = 1.0 - cosmo_dict["Omega_m"]
            Hubble = 100.0 * np.sqrt(
                cosmo_dict["Omega_m"] * (1.0 + redshift) ** 3 + Omega_l
            )
            rsd_factor = (1.0 + redshift) / Hubble
            pos[:, los] = pos[:, los] + vel[:, los] * rsd_factor
        return cls(
            galaxies_pos=pos,
            redshift=redshift,
            cosmo_dict=cosmo_dict,
            boxsize=boxsize,
            name=f"quijote_node{node}",
            n_mesh=n_mesh,
        )

    @classmethod
    def from_ascii(
        cls,
        filename,
        boxsize=1000.0,
        cosmo_dict={
            "Omega_m": 0.3175,
            "Omega_b": 0.049,
            "h": 0.6711,
            "ns": 0.9624,
            "sigma8": 0.834,
        },
        name=None,
        redshift: float = 0.0,
        los: Optional[int] = 2,
        n_mesh: Optional[int] = 360,
        min_halo_mass=1.0e12,
    ):
        import pandas as pd

        df = pd.read_csv(filename, skiprows=range(1, 19), header=0, sep=" ")
        pos = np.array(df[["x", "y", "z"]])
        vel = np.array(df[["vx", "vy", "vz"]])
        mass = np.array(df["m200c"])
        if min_halo_mass is not None:
            pos = pos[mass > min_halo_mass]
            vel = vel[mass > min_halo_mass]
            mass = mass[mass > min_halo_mass]
        if los is not None:
            Omega_l = 1.0 - cosmo_dict["Omega_m"]
            Hubble = 100.0 * np.sqrt(
                cosmo_dict["Omega_m"] * (1.0 + redshift) ** 3 + Omega_l
            )
            rsd_factor = (1.0 + redshift) / Hubble
            pos[:, los] = pos[:, los] + vel[:, los] * rsd_factor
        return cls(
            galaxies_pos=pos,
            redshift=redshift,
            cosmo_dict=cosmo_dict,
            name=name,
            n_mesh=n_mesh,
            boxsize=boxsize,
        )

    def to_nbodykit_catalogue(
        self,
        weights: Optional[np.array] = None,
    ) -> "nblab.ArrayCatalog":
        """Get a nbodykit catalogue from the catalogue

        Returns:
            nblab.ArrayCatalog: nbodykit catalogue
        """
        data = {
            "Position": self.galaxies_pos,
        }
        if weights is not None:
            data['Weights'] = weights
        return nblab.ArrayCatalog(
            data,
            BoxSize=self.boxsize,
            dtype=np.float32,
        )

    def to_mesh(
        self,
        n_mesh: int,
        resampler: str = "tsc",
        weights: Optional[np.array] = None,
        compensated: bool =True,
    ) -> np.array:
        """Get a mesh from the catalogue

        Args:
            n_mesh (int): number of cells in the mesh
            resampler (str, optional): resampler to use. Defaults to "tsc".

        Returns:
            np.array: mesh
        """
        nblab_cat = self.to_nbodykit_catalogue(weights=weights,)
        return nblab_cat.to_mesh(
            Nmesh=n_mesh,
            resampler=resampler,
            weight='Weights' if weights is not None else None,
            compensated=compensated,
        )

class SurveyCatalogue(BaseCatalogue):
    def __init__(
        self,
        galaxies_ra_dec_z: np.array,
        randoms_ra_dec_z: np.array,
        redshift: float,
        galaxies_nbar: float,
        randoms_nbar: float,
        fiducial_cosmology: "nblab.cosmology.Cosmology",
        weights: Optional[np.array] = None,
        name: Optional[str] = None,
        n_mesh: Optional[int] = 360,
    ):
        """Catalogue for survey data

        Args:
            galaxies_ra_dec_z (np.array): galaxies positions in ra, dec, z
            randoms_ra_dec_z (np.array): randoms positions in ra, dec, z
            redshift (float): mean redshift of the catalogue
            galaxies_nbar (float): number density of galaxies
            randoms_nbar (float): number density of randoms
            fiducial_cosmology (nblab.cosmology.Cosmology): fiducial cosmology, used to translate ra,dec,z into x,y,z
            weights (Optional[np.array], optional): galaxy weights, shape (N_galaxies,). Defaults to None.
            name (Optional[str], optional): catalogue name. Defaults to None.
            n_mesh (Optional[int], optional): number of cells in the mesh. Defaults to 360
        """
        galaxies_pos = self.sky_to_xyz(
            galaxies_ra_dec_z,
        )
        self.randoms_pos = self.sky_to_xyz(
            randoms_ra_dec_z,
        )
        self.weights = weights
        self.fiducial_cosmology = fiducial_cosmology
        self.galaxies_nbar = galaxies_nbar
        self.randoms_nbar = randoms_nbar
        super().__init__(
            self,
            galaxies_pos=galaxies_pos,
            redshift=redshift,
            name=name,
            n_mesh=n_mesh,
            boxsize=None,
        )

    @classmethod
    def from_boss(
        cls,
        galaxies_path: Path,
        randoms_path: Path,
        node: Optional[int] = None,
        weights: Optional[np.array] = None,
        mean_redshift: float = 0.5,
        name: str = None,
        n_mesh: Optional[int] = 360,
    ):
        """Get a catalogue for the BOSS survey

        Args:
            galaxies_path (Path): path to galaxy positions in ra,dec,z
            randoms_path (Path): path to randoms positions in ra,dec,z
            node (Optional[int], optional): node to read if simulations. Defaults to None.
            weights (Optional[np.array], optional):  weights for galaxies. Defaults to None.
            mean_redshift (float, optional): mean redshift of the catalogue. Defaults to 0.5.
            name (str, optional): catalogue name. Defaults to None.
            n_mesh (Optional[int], optional): number of cells in the mesh. Defaults to 360

        Returns:
            SurveyCatalogue
        """

        from cmass.survey.tools import BOSS_area
        from cmass.cmass.tools import get_nofz

        galaxies_ra_dec_z = np.load(galaxies_path / f"rdz{node}.npy")
        randoms_ra_dec_z = np.load(
            randoms_path / "random0_DR12v5_CMASS_North_PRECOMPUTED.npy"
        )
        fiducial_cosmology = nblab.cosmology.Planck15
        if weights is None:
            weights = np.ones(len(galaxies_ra_dec_z))
        fsky = BOSS_area() / (360.0**2 / np.pi)
        ng_of_z = get_nofz(galaxies_ra_dec_z[:, -1], fsky, cosmo=fiducial_cosmology)
        galaxies_nbar = ng_of_z(galaxies_ra_dec_z[:, -1])
        randoms_nbar = ng_of_z(randoms_ra_dec_z[:, -1])
        return cls(
            galaxies_ra_dec_z=galaxies_ra_dec_z,
            randoms_ra_dec_z=randoms_ra_dec_z,
            weights=weights,
            redshift=mean_redshift,
            galaxies_nbar=galaxies_nbar,
            randoms_nbar=randoms_nbar,
            fiducial_cosmology=fiducial_cosmology,
            name=name,
            n_mesh=n_mesh,
        )

    def sky_to_xyz(
        self,
        ra_dec_z: np.array,
    ) -> np.array:
        """Convert sky coordinates into x,y,z using the fiducial cosmology

        Args:
            ra_dec_z (np.array): ra, dec, z of shape (N_tracers, 3)

        Returns:
            np.array:  x,y,z of shape (N_tracers, 3)
        """
        return nblab.transform.SkyToCartesian(*ra_dec_z.T, self.fiducial_cosmology)

    def to_nbodykit_catalogue(self, weights=None, P0=1.0e4) -> "nblab.ArrayCatalog":
        """Get a nbodykit catalogue from the catalogue

        Args:
            P0 (float, optional): P0 for FKP weights. Defaults to 1.e4.

        Returns:
            nblab.ArrayCatalog: nbodykit catalogue
        """
        data = {
            "Position": self.galaxies_pos,
            "Nz": self.galaxies_nbar,
            "Weights": weights,
            "Weight_FKP": 1.0 / (1.0 + self.galaxies_nbar * P0),
        }
        randoms = {
            "Position": self.randoms_pos,
            "Nz": self.randoms_nbar,
            "Weights": np.ones(self.randoms_pos),
            "Weight_FKP": 1.0 / (1.0 + self.randoms_nbar * P0),
        }
        galaxies = nblab.ArrayCatalog(
            data,
            dtype=np.float32,
        )
        randoms = nblab.ArrayCatalog(
            data,
            dtype=np.float32,
        )
        return nblab.FKPCatalog(galaxies, randoms)

    def to_mesh(
        self,
        n_mesh: int,
        weights: Optional[np.array]=None, 
        resampler: str = "tsc",
    ) -> np.array:
        """Get a mesh from the catalogue

        Args:
            n_mesh (int): number of cells in the mesh
            weights (Optional[np.array], optional): weights for the tracers. Defaults to None.
            resampler (str, optional): resampler to use. Defaults to "tsc".

        Returns:
            np.array: mesh
        """
        nlab_cat = self.to_nbodykit_catalogue(weights=weights)
        return nlab_cat.to_mesh(
            Nmesh=n_mesh,
            nbar="Nz",
            fkp_weight="Weight_FKPla",
            weight="Weight",
            #com_weight = "Com_Weight"
            window=resampler,
        )
