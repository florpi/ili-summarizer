import numpy as np
import xarray as xr
from typing import List, Union
from summarizer.data import BoxCatalogue, SurveyCatalogue
from summarizer.base import BaseSummary
from pypower.fft_power import MeshFFTPower



class Pk(BaseSummary):
    def __init__(
        self,
        n_mesh: int = 360,
        ells: List[int] = [0, 2, 4],
        los: str = 'z',
        compensations: str = 'tsc',
        k_edges: Union[List, np.array] = None,
    ):
        """Compute two point power spectrum (in fourier space),
        using nbodykit

        Args:
            n_grid (int): compute the density field on a regular grid with grid x grid x grid voxels
            dk (float): spacing in k bins.
            ells (List[int]): list of multipoles to compute
            LOS (List[int]): line of sight (x,y,z)
        """
        self.n_mesh = n_mesh 
        self.ells = ells
        self.los = los 
        self.compensations = compensations
        self.k_edges = k_edges

    def __str__(self,):
        return 'pk'

    def __call__(
        self,
        catalogue: Union[BoxCatalogue, SurveyCatalogue],
        return_dataset: bool = False,
    ) -> np.array:
        """Given a catalogue, compute its power spectrum

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array:
        """
        galaxies_mesh = catalogue.get_mesh(self.n_mesh)
        if catalogue.is_periodic_box:
            power = MeshFFTPower(
                galaxies_mesh, 
                ells=self.ells, 
                los=self.los, 
                compensations=self.compensations,
                edges = self.k_edges,
            ).poles
        else:
            power = MeshFFTPower(
                galaxies_mesh, 
                ells=self.ells, 
                los='firstpoint',
                compensations=self.compensations,
                edges = self.k_edges,
            ).poles
        if return_dataset:
            return self.to_dataset(power)
        return power 

    def to_dataset(self, summary: np.array) -> xr.DataArray:
        """Convert a power spectrum array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        k = summary.k
        pk = []
        for ell in self.ells:
            pk.append(summary(ell=ell, complex=False, remove_shotnoise=True if ell == 0 else False))
        return xr.DataArray(
            np.vstack(pk),
            dims=("ells", "k"),
            coords={
                "ells": self.ells,
                "k": k,
            },
        )
