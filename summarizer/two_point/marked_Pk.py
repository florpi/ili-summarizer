import numpy as np
import xarray as xr
from typing import List, Optional
from summarizer.data import Catalogue
from summarizer.base import BaseSummary
from summarizer.utils import compute_overdensity
from nbodykit.filters import TopHat
from nbodykit.algorithms.fftpower import FFTPower

class Mk(BaseSummary):
    def __init__(
        self,
        density_radius: Optional[float]=30.,
        mark_delta_s: Optional[float] = 0.1,
        mark_p: Optional[float] = 1.,
        n_grid: int = 360,
        dk: float = 0.005,
        ells: List[int] = [0, 2, 4],
        LOS: List[int] = [0, 0, 1],

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
        self.density_radius = density_radius
        self.mark_delta_s = mark_delta_s
        self.mark_p = mark_p
        self.n_grid = n_grid
        self.dk = dk
        self.ells = ells
        self.LOS = LOS

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
        catalogue: Catalogue,
    )->np.array:
        """ Given a catalogue, compute its marked power spectrum.

        Args:
            catalogue (Catalogue): catalogue to summarize 

        Returns:
            np.array: marked power spectrum 
        """
        delta_env = compute_overdensity(
            catalogue.pos,
            tracers_mesh = catalogue.mesh,
            filter=TopHat(r=self.density_radius),
        )
        marked_mesh = catalogue.to_mesh(
            n_mesh = self.n_grid,
            weights=self.get_mark(delta=delta_env),
        )
        pk_moments = FFTPower(
            marked_mesh,
            mode="2d",
            dk=self.dk,
            kmin=0.0,
            poles=self.ells,
            los=self.LOS,
        )
        k = pk_moments.poles["k"]
        pks = []
        for ell in self.ells:
            multipole = pk_moments.poles[f"power_{ell}"].real
            #if ell == 0:
            #    multipole -= pk_moments.attrs["shotnoise"]
            # Issue with multiple shotnoise 
            pks.append(multipole)
        pks.append(k)
        return np.vstack(pks)    

    def to_dataset(self, summary: np.array) -> xr.DataArray:
        """Convert a power spectrum array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        return xr.DataArray(
            summary[:-1],
            dims=("ells", "k"),
            coords={
                "ells": self.ells,
                "k": summary[-1],
            },
        )
