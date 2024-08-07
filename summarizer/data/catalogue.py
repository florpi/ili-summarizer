import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union
from abc import abstractmethod

from pypower import CatalogMesh
from astropy import cosmology
from astropy.coordinates import SkyCoord
from astropy import units


class BaseCatalogue:
    def __init__(
        self,
        galaxies_pos: np.array,
        redshift: float,
        galaxies_weights: Optional[np.array] = None,
        boxsize: Optional[float] = None,
        name: Optional[str] = None,
        n_mesh: Optional[int] = 360,
        mesh_resampler: str= 'tsc',
    ):
        """Base catalogue object for manipulating pointclouds

        Args:
            galaxies_pos (np.array): 3D positions x,y,z, of shape
                (N_tracers, 3) in Mpc/h
            redshift (float): redshift of the catalogue
            galaxies_weights (Optional[np.array], optional): weights for the
                tracers. Defaults to None.
            boxsize (Optional[float], optional): simulation box size, None if
                not periodic cubic box. Defaults to None.
            name (Optional[str], optional): catalogue name. Defaults to None.
            n_mesh (Optional[int], optional): number of cells in mesh.
                Defaults to 360.
        """

        self.galaxies_pos = galaxies_pos
        if galaxies_weights is None:
            self.galaxies_weights = np.ones(len(galaxies_pos))
        else:
            self.galaxies_weights = galaxies_weights
        self.boxsize = boxsize
        self.name = name
        self.redshift = redshift
        self.n_mesh = n_mesh
        self.mesh_resampler = mesh_resampler
        if self.n_mesh is not None:
            self.galaxies_mesh = self.to_mesh(
                n_mesh=n_mesh,
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
        raise NotImplementedError('Meshing for BaseCatalog not implemented.')

    @classmethod
    def convert_cosmology(cls, cosmology_dict):
        return cosmology.FlatLambdaCDM(
                H0=cosmology_dict['h']*100,
                Om0=cosmology_dict['Omega_m'],
                Ob0=cosmology_dict['Omega_b']
            )

    @property
    def is_periodic_box(self,):
        return True if self.boxsize is not None else False

    @property
    def has_randoms(self,):
        return True if hasattr(self, 'randoms_pos') else False


    def get_mesh(self, n_mesh):
        if hasattr(self, "galaxies_mesh"):
            if self.galaxies_mesh.shape == (
                n_mesh,
                n_mesh,
                n_mesh,
            ):
                return self.galaxies_mesh
        self.galaxies_mesh = self.to_mesh(
            n_mesh=n_mesh,
        )
        return self.galaxies_mesh


class BoxCatalogue(BaseCatalogue):
    def __init__(
        self,
        galaxies_pos: np.array,
        boxsize: float,
        redshift: float,
        cosmology: Optional[cosmology.Cosmology] = None,
        galaxies_weights: Optional[np.array] = None,
        name: Optional[str] = None,
        n_mesh: Optional[int] = 360,
    ):
        """ Catalogue for a simulated box

        Args:
            galaxies_pos (np.array): galaxy positions x,y,z
            boxsize (float):  simulation box size
            redshift (float): redshift of the catalogue
            weights (Optional[np.array], optional): weights for the tracers.
                Defaults to None.
            cosmology (Dict[str, float]): true cosmology of the simulation
            name (Optional[str], optional): catalogue name. Defaults to None.
            n_mesh (Optional[int], optional): number of cells in mesh.
                Defaults to 360.
        """
        super().__init__(
            galaxies_pos=galaxies_pos,
            redshift=redshift,
            name=name,
            galaxies_weights=galaxies_weights,
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
    ) -> "BaseCatalogue":
        """Get a catalogue for the quijote simulations latin hyper cube

        Args:
            node (int): node to read
            redshift (float): redshift, one of 0.0, 0.5, 1.0, 2.0, 3.0
            n_halos (Optional[int], optional): Number of halos to include.
                Defaults to None.
            n_density_halos (Optional[int], optional): Number density of halos
                to select. Defaults to None.
            path_to_lhcs (Path, optional): Path to latin hypercube data.
            mesh (bool, optional): whether to create a mesh. Defaults to True.
            n_mesh (Optional[int], optional): number of cells in the mesh.
                Defaults to 50.

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
        pos %= boxsize
        cosmology = cls.convert_cosmology(
            cosmo_dict,
        )
        return cls(
            galaxies_pos=pos,
            redshift=redshift,
            cosmology=cosmology,
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
            cosmology=cosmo_dict,
            name=name,
            n_mesh=n_mesh,
            boxsize=boxsize,
        )

    def to_mesh(
        self,
        n_mesh: int,
        field: str = None,
        compensated: bool = True,
        weights: Optional[np.array] = None,
    ) -> np.array:
        """Get a mesh from the catalogue

        Args:
            n_mesh (int): number of cells in the mesh

        field : string, default=None
            Field to paint to mesh, one of:

            - "data": data positions and weights
            - "shifted": shifted positions and weights
                (available only if shifted positions are provided)
            - "randoms": randoms positions and weights
            - "data-normalized_shifted": shifted positions and weights,
                renormalized (by alpha) such that their sum is same as data
                weights
            - "data-normalized_randoms": randoms positions and weights,
                renormalized (by alpha) such that their sum is same as data
                weights
            - "fkp": FKP field, i.e. data - alpha * (shifted if provided
               else randoms)
            - ``None``: defaults to "data" if no shifted/randoms,
               else "fkp"

        resampler (str, optional): resampler to use. Defaults to "tsc".

        Returns:
            np.array: mesh
        """
        data_weights = np.ones(len(self.galaxies_pos))
        if self.galaxies_weights is not None:
            data_weights *= self.galaxies_weights
        if weights is not None:
            data_weights *= weights
        return CatalogMesh(
            data_positions=self.galaxies_pos,
            data_weights=data_weights,
            nmesh=n_mesh,
            resampler=self.mesh_resampler,
            position_type='pos',
            boxsize=self.boxsize,
            boxcenter=[self.boxsize / 2.0, self.boxsize / 2.0, self.boxsize / 2.0],
            wrap=True,
        ).to_mesh(field=field, dtype=np.float32, compensate=compensated)


class SurveyCatalogue(BaseCatalogue):
    def __init__(
        self,
        galaxies_ra_dec_z: np.array,
        randoms_ra_dec_z: np.array,
        redshift: float,
        fiducial_cosmology:
            Union[Dict[str, float], cosmology.Cosmology] = None,
        galaxies_weights: np.array = None,
        randoms_weights: np.array = None,
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
            fiducial_cosmology (dict, astropy.cosmology.Cosmology): fiducial
                cosmology, used to translate ra,dec,z into x,y,z. Acceptable
                as a dictionary of parameters or as an astropy cosmology.
                Defaults to Planck cosmology,
            weights (Optional[np.array], optional): galaxy weights, shape
                (N_galaxies,). Defaults to None.
            name (Optional[str], optional): catalogue name. Defaults to None.
            n_mesh (Optional[int], optional): number of cells in the mesh.
                Defaults to 360
        """
        self.fiducial_cosmology = fiducial_cosmology
        galaxies_pos = self.sky_to_xyz(galaxies_ra_dec_z)
        self.randoms_pos = self.sky_to_xyz(randoms_ra_dec_z)
        self.galaxies_weights = galaxies_weights
        self.randoms_weights = randoms_weights
        super().__init__(
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
        galaxies_weights: Optional[np.array] = None,
        mean_redshift: float = 0.5,
        name: str = None,
        n_mesh: Optional[int] = 360,
    ):
        """Get a catalogue for the BOSS survey

        Args:
            galaxies_path (Path): path to galaxy positions in ra,dec,z
            randoms_path (Path): path to randoms positions in ra,dec,z
            node (Optional[int], optional): node to read if simulations.
                Defaults to None.
            weights (Optional[np.array], optional):  weights for galaxies.
                Defaults to None.
            mean_redshift (float, optional): mean redshift of the catalogue.
                Defaults to 0.5.
            name (str, optional): catalogue name. Defaults to None.
            n_mesh (Optional[int], optional): number of cells in the mesh.
                Defaults to 360

        Returns:
            SurveyCatalogue
        """

        galaxies_ra_dec_z = np.load(galaxies_path / f"rdz{node}.npy")
        randoms_ra_dec_z = np.load(
            randoms_path / "random0_DR12v5_CMASS_North_PRECOMPUTED.npy"
        )
        fiducial_cosmology = cosmology.Planck15
        if galaxies_weights is None:
            galaxies_weights = np.ones(len(galaxies_ra_dec_z))
        return cls(
            galaxies_ra_dec_z=galaxies_ra_dec_z,
            randoms_ra_dec_z=randoms_ra_dec_z,
            galaxies_weights=galaxies_weights,
            redshift=mean_redshift,
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

        ra, dec, z = ra_dec_z.T
        pos = SkyCoord(ra=ra*units.deg, dec=dec*units.deg,
                       distance=self.fiducial_cosmology.comoving_distance(z))
        pos = pos.cartesian.xyz
        pos *= self.fiducial_cosmology.h  # convert from Mpc to Mpc/h

        return pos.value.T

    def to_mesh(
        self,
        n_mesh: int,
        field: str = None,
        resampler: str = "tsc",
        compensated: bool = True,
        weights: Optional[np.array] = None,
    ) -> np.array:
        """Get a mesh from the catalogue

        Args:
            n_mesh (int): number of cells in the mesh

        field : string, default=None
            Field to paint to mesh, one of:

            - "data": data positions and weights
            - "shifted": shifted positions and weights
                (available only if shifted positions are provided)
            - "randoms": randoms positions and weights
            - "data-normalized_shifted": shifted positions and weights,
                renormalized (by alpha) such that their sum is same as data
                weights
            - "data-normalized_randoms": randoms positions and weights,
                renormalized (by alpha) such that their sum is same as data
                weights
            - "fkp": FKP field, i.e. data - alpha * (shifted if provided
               else randoms)
            - ``None``: defaults to "data" if no shifted/randoms,
               else "fkp"

        resampler (str, optional): resampler to use. Choices are
            ['ngp', 'cic', 'tcs', 'pcs']. Defaults to "tsc".

        Returns:
            np.array: mesh
        """
        data_weights = np.ones(len(self.galaxies_pos))
        if self.galaxies_weights is not None:
            data_weights *= self.galaxies_weights
        if weights is not None:
            data_weights *= weights
        return CatalogMesh(
            data_positions=self.galaxies_pos,
            data_weights=data_weights,
            randoms_positions=self.randoms_pos,
            randoms_weights=self.randoms_weights,
            nmesh=n_mesh,
            resampler=resampler,
            position_type='pos',
            boxsize=self.boxsize,
            wrap=False
        ).to_mesh(field=field, dtype=np.float32, compensate=compensated)
