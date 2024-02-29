import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Union
import nbodykit
from nbodykit import cosmology

try:
    import nbodykit.lab as nblab 
except ModuleNotFoundError:
    print('failed nbodykit.lab loading!')

    pass
import os
from os.path import join as pjoin



from ltu-cmass.cmass.tools import (get_nofz, sky_to_xyz, load_galaxies_obs,
                    load_randoms_precomputed)
from ltu-cmass.survey.tools import BOSS_area
# from ltu-cmass.utils import get_source_path, timing_decorator
from nbodykit import cosmology

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


class Catalogue:
    def __init__(
        self,
        pos: np.array,
        # vel: np.array, #make this optional?
        # redshift: float,
        # boxsize: float,
        # cosmo_dict: Dict[str, float],
        # name: str,
        # mass: Optional[np.array] = None,
        # mesh: bool = True,
        # n_mesh: Optional[int] = 360,
        vel: np.array = None,
        redshift: float = None,
        boxsize: float = None,
        cosmo_dict: Dict[str, float] = None,
        name: str = '',
        mass: Optional[np.array] = None,
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
        self.mass = mass
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
        mesh: bool = True,
        n_mesh: Optional[int] = 360,
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
        boxsize = 1000.0
        if n_halos is None and n_density_halos is not None:
            n_halos = int(n_density_halos * boxsize **3)
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
            rsd_factor = (1.0 + redshift)/Hubble
            pos[:, los] = pos[:, los] + vel[:, los] * rsd_factor
        return cls(
            pos=pos,
            vel=vel,
            redshift=redshift,
            cosmo_dict=cosmo_dict,
            boxsize=boxsize,
            name=f'quijote_node{node}',
            mass=mass,
            mesh=mesh,
            n_mesh=n_mesh,
        )


    @classmethod
    def CMASS_proc(
        cls, #WHAT IS THISSSSSSS apparently part of classmethod?
        cfg: str,
        node: int,
        mesh: bool = True,
        n_mesh: Optional[int] = 360,
        weights = None
    ) -> "Catalogue":
        """Get a catalogue from the CMASS-post-processed latin hyper cube, for some HOD variation

        Args:
            node (int): node/which LH number to read 
            mesh (bool, optional): whether to create a mesh. Defaults to True.
            n_mesh (Optional[int], optional): number of cells in the mesh. Defaults to 50.

        Returns:
            Catalogue: catalogue for simulation
        """
        def load_cmass_sim(cfg):
            source_dir = get_source_path(cfg, cfg.sim)
            rdz_sim = np.load(pjoin(source_dir, 'obs', f'rdz{cfg.bias.hod.seed}.npy')) 
            return rdz_sim #RA, dec, redshift (comes from position in box AND peculiar velocity)
        rdz_sim=load_cmass_sim(cfg)
        # cosmo_dict = load_params_sim(node=node, path_to_lhcs=path_to_lhcs) 
        cosmo_dict = cosmology.Planck15  # fixed because we don't know true cosmology
        #assumed in Pk_cmass that don't know real life cosmo, vs real data, only used to calculate comoving distance to the thing, should still reflect it eventually?
        gpos = nblab.transform.SkyToCartesian(*rdz_sim.T, cosmo_dict)

        if weights is None:
            weights = np.ones(len(rdz_sim))

        #next: add the weights thing? comp vs fkp weights? the comp weights as above will work fine
        #don't have velocity, can't do velocity field OR masses... hmmm. but do have redshift
        #want positions as comoving
        return cls(
                pos=gpos,
                cosmo_dict=cosmo_dict,
                name=f'{cfg.sim}_node{node}_hod{cfg.bias.hod.seed}',
                mesh=mesh,
                n_mesh=n_mesh,
                weights=weights
            )

    def to_nbodykit_catalogue(self,weights=None)->"nblab.ArrayCatalog":
        """ Get a nbodykit catalogue from the catalogue

        Returns:
            nblab.ArrayCatalog: nbodykit catalogue 
        """
        if weights is not None:
            data =  {'Position': self.pos, 'Weights': weights, 'Mass': self.mass, 'Velocity': self.vel} 
        else:
            data =  {'Position': self.pos, 'Mass': self.mass, 'Velocity': self.vel} 
        return nblab.ArrayCatalog(
                data,
                BoxSize=self.boxsize, 
                dtype=np.float32, 
            ) 

    def to_nbodykit_halo_catalogue(self,)->"nblab.HaloCatalog":
        """ Get a nbodykit catalogue from the catalogue

        Returns:
            nblab.ArrayCatalog: nbodykit catalogue 
        """
        cat = self.to_nbodykit_catalogue()
        Mnu = self.cosmo_dict['Mnu'] if 'Mnu' in self.cosmo_dict else 0.0
        cosmo = nblab.cosmology.Planck15.clone(
            h=self.cosmo_dict['h'], 
            Omega0_b=self.cosmo_dict['Omega_b'], 
            Omega0_cdm=self.cosmo_dict['Omega_m'] - self.cosmo_dict['Omega_b'],
            m_ncdm=[None, Mnu][Mnu>0.],
            n_s=self.cosmo_dict['n_s'],
        ) 
        return nblab.HaloCatalog(
            cat, cosmo=cosmo, redshift=self.redshift, mdef='vir',
        ) 

    def to_mesh(self, n_mesh: int, resampler: str = "tsc", weights=None,) -> np.array:
        """Get a mesh from the catalogue

        Args:
            n_mesh (int): number of cells in the mesh
            resampler (str, optional): resampler to use. Defaults to "tsc".

        Returns:
            np.array: mesh
        """
        nblab_cat = self.to_nbodykit_catalogue(
            weights=weights
        )
        if weights is not None:
            return nblab_cat.to_mesh(
                Nmesh=n_mesh, 
                resampler=resampler,
                weight='Weights'
            )
        return nblab_cat.to_mesh(
            Nmesh=n_mesh, 
            resampler=resampler,
        )

'''what does the catalogue object have?
it can send itself to a mesh
it can give a halo-only catalog as an nblab.HaloCatalog; takes no weights, ties cosmo to it clearly?
it can give a (general) catalogue as an nblab.ArrayCatalog, can take in weights
'''



# #how to import functions from a different path? just do a symbolic link maybe? hmmmmmmm
#   
# class FPKCatalogue:
#     def __init__(
#         self,
#         pos: np.array,
#         vel: np.array,
#         redshift: float,
#         boxsize: float,
#         cosmo_dict: Dict[str, float],
#         name: str,
#         mass: Optional[np.array] = None,
#         mesh: bool = True,
#         n_mesh: Optional[int] = 360,
#     ):
#         """Catalogue of tracers (dark matter halos, galaxies...)

#         Args:
#             pos (np.array): 3D vector of positions, size (N, 3) 
#             vel (np.array): 3D vector of velocities, size (N, 3) 
#             redshift (float): redshift of the catalogue
#             boxsize (float): size of the simulation box
#             cosmo_dict (Dict[str, float]): dictionary fo cosmological parameters
#             mesh (bool, optional): whether to create a mesh. Defaults to True.
#             n_mesh (Optional[int], optional): number of cells in the mesh. Defaults to 50.
#         """
#         self.pos = pos % boxsize  
#         self.vel = vel
#         self.mass = mass
#         self.redshift = redshift
#         self.boxsize = boxsize
#         self.cosmo_dict = cosmo_dict
#         self.name = name
#         if mesh:
#             self.mesh = self.to_mesh(n_mesh=n_mesh)

#     def __str__(self,)->str:
#         """get name for catalogue

#         Returns:
#             str: name
#         """
#         return self.name

#     def __len__(
#         self,
#     ) -> int:
#         """ Get number of objects in catalogue

#         Returns:
#             int: n tracers
#         """
#         return len(self.pos)

#     @classmethod
#     def from_quijote(
#         cls,
#         node: int,
#         redshift: float,
#         path_to_lhcs: Union[Path,str], 
#         n_halos: Optional[int] = None,
#         n_density_halos: Optional[float] = None,
#         los: Optional[int] = 2,
#         mesh: bool = True,
#         n_mesh: Optional[int] = 360,
#     ) -> "Catalogue":
#         """Get a catalogue for the quijote simulations latin hyper cube

#         Args:
#             node (int): node to read
#             redshift (float): redshift, one of 0.0, 0.5, 1.0, 2.0, 3.0
#             n_halos (Optional[int], optional): Number of halos to include. Defaults to None.
#             n_density_halos (Optional[int], optional): Number density of halos to select. Defaults to None.
#             path_to_lhcs (Path, optional): Path to latin hypercube data. 
#             mesh (bool, optional): whether to create a mesh. Defaults to True.
#             n_mesh (Optional[int], optional): number of cells in the mesh. Defaults to 50.

#         Returns:
#             Catalogue: catalogue for simulation
#         """
#         from summarizer.data.quijote_utils import load_params_sim, load_sim

#         path_to_lhcs = Path(path_to_lhcs)
#         pos, vel, mass = load_sim(
#             node=node, redshift=redshift, path_to_lhcs=path_to_lhcs
#         )
#         boxsize = 1000.0
#         if n_halos is None and n_density_halos is not None:
#             n_halos = int(n_density_halos * boxsize **3)
#         if n_halos is not None:
#             sorted_mass_idx = np.argsort(mass)
#             pos = pos[sorted_mass_idx][-n_halos:, :]
#             mass = mass[sorted_mass_idx][-n_halos:]
#             vel = vel[sorted_mass_idx, :][-n_halos:, :]
#         cosmo_dict = load_params_sim(node=node, path_to_lhcs=path_to_lhcs)
#         if los is not None:
#             Omega_l = 1.0 - cosmo_dict["Omega_m"]
#             Hubble = 100.0 * np.sqrt(
#                 cosmo_dict["Omega_m"] * (1.0 + redshift) ** 3 + Omega_l
#             )
#             rsd_factor = (1.0 + redshift)/Hubble
#             pos[:, los] = pos[:, los] + vel[:, los] * rsd_factor
#         return cls(
#             pos=pos,
#             vel=vel,
#             redshift=redshift,
#             cosmo_dict=cosmo_dict,
#             boxsize=boxsize,
#             name=f'quijote_node{node}',
#             mass=mass,
#             mesh=mesh,
#             n_mesh=n_mesh,
#         )
    
#     # def to_nbodykit_catalogue(self,weights=None)->"nblab.ArrayCatalog":
#     #     """ Get a nbodykit catalogue from the catalogue

#     #     Returns:
#     #         nblab.ArrayCatalog: nbodykit catalogue 
#     #     """
#     #     if weights is not None:
#     #         data =  {'Position': self.pos, 'Weights': weights, 'Mass': self.mass, 'Velocity': self.vel} 
#     #     else:
#     #         data =  {'Position': self.pos, 'Mass': self.mass, 'Velocity': self.vel} 
#     #     return nblab.ArrayCatalog(
#     #             data,
#     #             BoxSize=self.boxsize, 
#     #             dtype=np.float32, 
#     #         ) 
#     def to_nbodykit_gal_catalogue(self)->"nblab.ArrayCatalog":
#         """ Get a nbodykit catalogue from the catalogue

#         Returns:
#             nblab.ArrayCatalog: nbodykit catalogue 
#         """
#         # _gals = nblab.ArrayCatalog({
#         # 'Position': gpos,
#         # 'NZ': nbar_g,
#         # 'WEIGHT': weights,
#         # 'WEIGHT_FKP': 1./(1. + nbar_g * P0)
#         #      })
#         P0 = 1e4

#         source_path = get_source_path(cfg, cfg.sim)
#         grdz = load_galaxies_obs(source_path, cfg.bias.hod.seed)
#         rrdz = load_randoms_precomputed()
#         gpos = sky_to_xyz(grdz, cosmo)
#         rpos = sky_to_xyz(rrdz, cosmo)
#         fsky = BOSS_area() / (360.**2 / np.pi)
#         ng_of_z = get_nofz(grdz[:, -1], fsky, cosmo=cosmo)
#         nbar_r = ng_of_z(rrdz[:, -1])
#         nbar_g = ng_of_z(grdz[:, -1])

#         data =  {'Position': gpos, 'Weights': self.weights, 'Nz': nbar_g, 'WeightFPK': 1./(1. + nbar_g * P0)} 
#         _gals=nblab.ArrayCatalog(data) 

#         rands =  {'Position': rpos, 'Weights': self.weights, 'Nz':  nbar_r, 'WeightFPK': 1./(1. + nbar_r * P0)} 
#         _rands=nblab.ArrayCatalog(rands) 
        
#         return _gals, _rands


#     def to_nbodykit_randoms_catalogue(self????,)->"nblab.HaloCatalog":
#         """ Get a nbodykit catalogue from the catalogue

#         Returns:
#             nblab.ArrayCatalog: nbodykit catalogue 
#         """
#         # cat = self.to_nbodykit_catalogue()
#         # Mnu = self.cosmo_dict['Mnu'] if 'Mnu' in self.cosmo_dict else 0.0
#         # cosmo = nblab.cosmology.Planck15.clone(
#         #     h=self.cosmo_dict['h'], 
#         #     Omega0_b=self.cosmo_dict['Omega_b'], 
#         #     Omega0_cdm=self.cosmo_dict['Omega_m'] - self.cosmo_dict['Omega_b'],
#         #     m_ncdm=[None, Mnu][Mnu>0.],
#         #     n_s=self.cosmo_dict['n_s'],
#         # ) 
#         # return nblab.HaloCatalog(
#         #     cat, cosmo=cosmo, redshift=self.redshift, mdef='vir',
#         # ) 


#     def to_FPKmesh(gal_cat, rand_cat, n_mesh: int, resampler: str = "tsc", weights=None,) -> np.array:
#         """Get a mesh from the catalogue

#         Args:
#             n_mesh (int): number of cells in the mesh, was Ngrid in Pk_cmass.py
#             resampler (str, optional): resampler to use. Defaults to "tsc". "window" in last functions

#         Returns:
#             np.array: mesh
#         """
#         _gal = gal_cat.to_nbodykit_catalogue(
#             weights=weights
#         )
#         _rands = rand_cat.to_nbodykit_catalogue(
#             weights=weights
#         )
#         fkp = nblab.FKPCatalog(_gals, _rands)
#         return fkp.to_mesh(Nmesh=n_mesh, nbar='NZ', fkp_weight='WEIGHT_FKP',
#                        comp_weight='WEIGHT', window=resampler
#                        )
#         # if weights is not None:
#         #     return nblab_cat.to_mesh(
#         #         Nmesh=n_mesh, 
#         #         resampler=resampler,
#         #         weight='Weights'
#         #     )
#         # return nblab_cat.to_mesh(
#         #     Nmesh=n_mesh, 
#         #     resampler=resampler,
#         # )
