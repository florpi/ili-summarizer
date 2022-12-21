import numpy as np
import xarray as xr
from typing import List, Union
# from pycorr import TwoPointCorrelationFunction
from summarizer.data import Catalogue
from summarizer.base import BaseSummary
import Corrfunc
from Corrfunc.theory.vpf import vpf

class VPF(BaseSummary):
    def __init__(
        self,
        rmax: List[int],
        nbins: List[int],
        nspheres: List[int],
        boxsize: List[int],
        numpN: int = 2,
        seed: int = 4,
        # X: List[int],
        # Y: List[int],
        # Z: List[int],
        max_cells_per_dim: int = 50,
        # periodic: boolean = True,
    ):
        """Compute the VPF using Corrfunc
        With current set up of how the quijote halo catalogs are read in, it will be
        in redshift space--be wary of this!!!! In the future, Lucia will hopefully code
        in the choice of either redshift vs. configuration space calculations
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

        Paco: halos are in kpc/h


        Args:
            rmax: double
                Maximum radius of the sphere to place on the particles

            nbins: integer
                Number of bins in the counts-in-cells. Radius of first shell
                is rmax/nbins

            nspheres: integer (>= 0)
                Number of random spheres to place within the particle distribution.
                For a small number of spheres, the error is larger in the measured
                pN's.

            numpN: integer (>= 1)
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

            seed: unsigned integer
                Random number seed for the underlying GSL random number generator. Used
                to draw centers of the spheres.

            X/Y/Z: arraytype, real (float/double)
                Particle positions in the 3 axes. Must be within [0, boxsize]
                and specified in the same units as ``rp_bins`` and boxsize. All
                3 arrays must be of the same floating-point type.

                Calculations will be done in the same precision as these arrays,
                i.e., calculations will be in floating point if XYZ are single
                precision arrays (C float type); or in double-precision if XYZ
                are double precision arrays (C double type).

            verbose: boolean (default false)
                Boolean flag to control output of informational messages

            periodic: boolean
                Boolean flag to indicate periodic boundary conditions.

            boxsize: double or 3-tuple of double, required if ``periodic=True``
                The (X,Y,Z) side lengths of the spatial domain. Present to facilitate
                exact calculations for periodic wrapping. A scalar ``boxsize`` will
                be broadcast to a 3-tuple. If the boxsize in a dimension is 0., then
                then that dimension's wrap is done based on the extent of the particle
                distribution. If the boxsize in a dimension is -1., then periodicity
                is disabled for that dimension.

                .. versionchanged:: 2.4.0
                   Required if ``periodic=True``.

                .. versionchanged:: 2.5.0
                   Accepts a 3-tuple of side lengths.

        """
        if type(rmax) is str:
            self.rmax = eval(rmax)
        else:
            self.rmax = np.array(rmax)

        if type(nbins) is str:
            self.nbins = eval(nbins)
        else:
            self.nbins = np.array(nbins)

        if type(nspheres) is str:
            self.nspheres = eval(nspheres)
        else:
            self.nspheres = np.array(nspheres)

        self.numpN = numpN
        self.seed = seed

        # if type(data_positions_X) is str:
        #     self.data_positions_X = eval(data_positions_X)
        # else:
        #     self.data_positions_X = np.array(data_positions_X)
        #
        # if type(data_positions_Y) is str:
        #     self.data_positions_Y = eval(data_positions_Y)
        # else:
        #     self.data_positions_Y = np.array(data_positions_Y)
        #
        # if type(data_positions_Z) is str:
        #     self.data_positions_Z = eval(data_positions_Z)
        # else:
        #     self.data_positions_Z = np.array(data_positions_Z)

        self.max_cells_per_dim = max_cells_per_dim
        # self.periodic = periodic

        if type(boxsize) is str:
            self.boxsize = eval(boxsize)
        else:
            self.boxsize = np.array(boxsize)

        # self.r = 0.5*(self.r_bins[1:] + self.r_bins[:-1])
        # if type(mu_bins) is str:
        #     self.mu_bins = eval(mu_bins)
        # else:
        #     self.mu_bins = np.array(mu_bins)
        # self.mu = 0.5*(self.mu_bins[1:] + self.mu_bins[:-1])
        # self.ells = ells
        # self.n_threads = n_threads

    def __call__(self, catalogue: Catalogue) -> np.array:
        """ Given a catalogue, compute its VPF for a set of radii/distance scales

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array: the void probability function, or probability that a sphere of a given radius is empty in the sample
        """
        return vpf(
            rmax=(self.rmax),
            nbins=(self.nbins),
            nspheres=(self.nspheres),
            numpN=(self.numpN),
            seed=(self.seed),
            X=catalogue.pos[:,0],
            Y=catalogue.pos[:,1],
            Z=catalogue.pos[:,2],
            max_cells_per_dim=self.max_cells_per_dim,
            periodic=True,
            boxsize=catalogue.boxsize,
        )#0(ells=self.ells)


    def to_dataset(self, summary: np.array)->xr.DataArray:
        """ Convert a tpcf array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        print(summary.shape, summary)
        Radii=np.zeros(len(summary))
        VPFs=np.zeros(len(summary))
        for A in range(0, len(summary),1):
            Radii[A] = summary[A][0]
            VPFs[A] = summary[A][1][0]
        print(Radii, Radii.shape)
        print(VPFs, VPFs.shape)
        final=np.stack((Radii, VPFs), axis=1)
        print(final, final.shape)
        
        return xr.DataArray(
            final,
            dims=('r', 'P0s'),
            coords = {
                # 'ells': self.ells,
                'r': Radii,
                'P0s': VPFs,
            },
        )
