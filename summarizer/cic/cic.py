import numpy as np
import xarray as xr
from typing import List
from summarizer.data import Catalogue
from summarizer.base import BaseSummary
from Corrfunc.theory.vpf import vpf

class CiC(BaseSummary):
    def __init__(
        self,
        r_max: float,
        n_bins: int,
        n_spheres: List[int],
        num_pN: int = 5,
        random_seed: int = 4,
        max_cells_per_dim: int = 50,
    ):
        """Compute the probability of finding N number of tracers inside a sphere
        of a given maximum radius. This is a generalization of the void probability.


        Args:
            r_max (float): Maximum radius of the sphere s
            n_bins (int): Number of bins in r. Radius of first shell is rmax/nbins
            nspheres (int): integer (>= 0)
                Number of random spheres to place within the particle distribution.
                For a small number of spheres, the error is larger in the measured
                pN's.
            num_pN (int): integer (>= 1)
                Governs how many unique pN's are to returned. If ``numpN`` is set to 1,
                then only the vpf (p0) is returned. For ``numpN=2``, p0 and p1 are
                returned.

            random_seed: unsigned integer
                Random number seed for the underlying GSL random number generator. Used
                to draw centers of the spheres.
        """
        self.r_max = r_max
        self.n_bins = n_bins
        self.n_spheres = n_spheres
        self.num_pN = num_pN
        self.random_seed = random_seed
        self.max_cells_per_dim = max_cells_per_dim    
        
    def __call__(self, catalogue: Catalogue) -> np.array:
        """ Given a catalogue, compute CiC up to numpN for a set of 
        radii/distance scales.

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array: the probability of finding N tracers inside random spheres
        """
        return vpf(
            rmax=self.r_max,
            nbins=self.n_bins,
            nspheres=self.n_spheres,
            numpN=self.num_pN,
            seed=self.random_seed,
            X=catalogue.pos[:,0],
            Y=catalogue.pos[:,1],
            Z=catalogue.pos[:,2],
            max_cells_per_dim=self.max_cells_per_dim,
            periodic=True,
            boxsize=catalogue.boxsize,
        )

    def to_dataset(self, summary: np.array)->xr.DataArray:
        """ Convert a tpcf array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        radii = [t[0] for t in summary]
        p_N = np.array([t[1] for t in summary])
        return xr.DataArray(
            p_N,
            dims=('r', 'N'),
            coords = {
                'r': radii,
                'N': np.arange(0, self.num_pN),
            },
        )
