import numpy as np
import xarray as xr
from typing import List, Union
from summarizer.data import Catalogue
from summarizer.base import BaseSummary
from Corrfunc.theory.vpf import vpf

class VPF(BaseSummary):
    def __init__(
        self,
        r_max: float,
        n_bins: int,
        n_spheres: List[int],
        num_pN: int = 2,
        random_seed: int = 4,
        max_cells_per_dim: int = 50,
    ):
        """Compute the VPF using Corrfunc
        Note that the VPF function by default does not take in a list of radii
        to calculate the VPF for--it takes the maximum radius and a number of bins
        to compute it (first is rmax/nbins), so be sure to know what radii it is
        being calculated for.
        Also, by default, Lucia has chosen the random seed of 4 to make sure her
        checks are reproducible. To check the VPF calculation, do this: feed it
        randomly generated positions, and compare the log10 of the output at
        various radii to: - rho * ( (4/3)*pi*R^3 ) / (ln(10)). (3D VPF) This is the
        theoretical 3D VPF of a random distribution, should work for both real and
        redshift space; with enough dropped spheres, the curves should be VERY
        close together.
        Guidance for the VPF, and clustering in general, partially from Perez et
        al 2021 section 3.3: Have at least 10,000 random points (or 50xnumber of
        galaxies, whichever is bigger), and make sure that they're randomly distributed
        according to the how the sample is (i.e. consider mask and completeness
        over the area). Smallest distance scale corresponds to the density of your
        sample; largest to the full area or volume you're covering and the precision
        you need for the VPF (log10VPF to between -2 to 4 has been enough for most
        of LAP's applications; for example, if you want to be able to measure the
        VPF to a value of 1/100, you need to be able to fit at least 100 INDEPENDENT
        circles or spheres among the area your galaxy covers; worry less about masks
        here, just total covered area, as a good starting point). Best way to get a
        sense of error is splitting up your sample into independent subsamples and
        measure the VPF on them, jackknife tends to somewhat underestimate the error.

        Args:
            r_max (float): Maximum radius of the sphere to place on the particles
            n_bins (int): Number of bins in the counts-in-cells. Radius of first shell
            is rmax/nbins
            nspheres (int): integer (>= 0)
                Number of random spheres to place within the particle distribution.
                For a small number of spheres, the error is larger in the measured
                pN's.
            num_pN (int): integer (>= 1)
                Governs how many unique pN's are to returned. If ``numpN`` is set to 1,
                then only the vpf (p0) is returned. For ``numpN=2``, p0 and p1 are
                returned.

                More explicitly, the columns in the results look like the following:

                 ======   ==========================
                 numpN    Columns in output
                 ======   ==========================
                    1      p0
                    2      p0      p1
                    3      p0      p1     p2
                    4      p0      p1     p2     p3
                 ======   ==========================

                and so on...

                Note: ``p0`` is the vpf

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
        """ Given a catalogue, compute its VPF for a set of radii/distance scales

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array: the void probability function, 
            or probability that a sphere of a given radius is empty in the sample
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
        p0 = [t[1][0] for t in summary]
        return xr.DataArray(
            p0,
            dims=('r'),
            coords = {
                'r': radii,
            },
        )
