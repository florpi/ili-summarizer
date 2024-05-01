import numpy as np
import xarray as xr
from typing import List, Optional, Union
from summarizer.data import BoxCatalogue, SurveyCatalogue
from summarizer.base import BaseSummary
from summarizer.utils import compute_overdensity
from pypower.fft_power import MeshFFTPower

class Mk(BaseSummary):
    def __init__(
        self,
        smoothing_radius: Optional[float]=30.,
        mark_delta_s: Optional[float] = 0.1,
        mark_p: Optional[float] = 1.,
        n_mesh: int = 360,
        ells: List[int] = [0, 2, 4],
        los: str = 'z',
        compensations: str = 'tsc',

    ):
        """ Compute the marked power spectrum as in arxiv:2001.11024, where the mark
        is defined by the tracer's environment.

        Args:
            density_radius (Optional[float], optional): radius used to define a top-hat filter to 
            estimate overdensity. Defaults to 30.
            mark_delta_s (Optional[float], optional): density parameter defining the mark. Defaults to 0.1.
            mark_p (Optional[float], optional): exponent in mark function. Defaults to 1..
            n_grid (int, optional): number of grid points in mesh used to estimate power 
            spectrum. Defaults to 360.
            dk (float): spacing in k bins.
            ells (List[int]): list of multipoles to compute
            LOS (List[int]): line of sight (x,y,z)
        """
        self.smoothing_radius = smoothing_radius 
        self.mark_delta_s = mark_delta_s
        self.mark_p = mark_p
        self.n_mesh =n_mesh 
        self.ells = ells
        self.los = los 
        self.compensations = compensations

    def __str__(self,):
        return 'mk'
    
    def get_mark(self, delta: np.array,)->np.array:
        """ Given the tracer's environment ```delta''', compute the mark function.

        Args:
            delta (np.array): tracer's environment. 

        Returns:
            np.array: value of mark 
        """
        return ((self.mark_delta_s+1.)/(self.mark_delta_s+ delta +1.0+1e-6))**self.mark_p

    def __call__(
        self,
        catalogue: Union[BoxCatalogue, SurveyCatalogue],
        return_dataset: bool = False,
    )->np.array:
        """ Given a catalogue, compute its marked power spectrum.

        Args:
            catalogue (Catalogue): catalogue to summarize 

        Returns:
            np.array: marked power spectrum 
        """
        overdensity = compute_overdensity(
            query_positions=catalogue.pos,
            catalogue=catalogue,
            smoothing_radius=self.smoothing_radius,
        )
        # get a mesh weighted by the density field
        marked_mesh = catalogue.to_mesh(
            n_mesh = self.n_grid,
            weights=self.get_mark(delta=overdensity),
        )
        if catalogue.is_periodic_box:
            power = MeshFFTPower(
                marked_mesh, 
                edges=None, 
                ells=self.ells, 
                los=self.los, 
                compensations=self.compensations,
            ).poles
        else:
            power = MeshFFTPower(
                marked_mesh, 
                edges=None, 
                ells=self.ells, 
                los='firstpoint',
                compensations=self.compensations,
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
