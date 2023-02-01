import numpy as np
import xarray as xr
from typing import List, Union
# from pycorr import TwoPointCorrelationFunction
from summarizer.data import Catalogue
from summarizer.base import BaseSummary
import Corrfunc
from Corrfunc.theory.vpf import vpf

class CiC(BaseSummary):
    def __init__(
        self,
        rmax: List[int],
        nbins: List[int],
        nspheres: List[int],
        boxsize: List[int],
        numpN: List[int],
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

                LAP: For most applications, numpN less than 500 will be more than enough!
                numpN=50 may even get everything you need. Nearby numpNs are likely correlated.
                Note: this will be SLOW, but has lots of information. Which numpN
                have useful information also depends on the number density of galaxies,
                as well as the measured distance scale. See Table 3 of Perez, Genel,
                et al 2022 to get a sense of which numpN have the most useful information
                for different scales for samples of a few thousand galaxies.

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

        if type(numpN) is str:
            self.numpN = eval(numpN)
        else:
            self.numpN = np.array(numpN)

        # self.numpN = numpN
        self.seed = seed

        self.max_cells_per_dim = max_cells_per_dim
        # self.periodic = periodic

        if type(boxsize) is str:
            self.boxsize = eval(boxsize)
        else:
            self.boxsize = np.array(boxsize)

    def __call__(self, catalogue: Catalogue) -> np.array:
        """ Given a catalogue, compute CiC up to numpN for a set of radii/distance scales.

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


    def to_dataset_old(self, summary: np.array)->xr.DataArray:
        """ Convert a tpcf array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        print(summary.shape, summary)
        Radii=np.zeros(len(summary))
        # VPFs=np.zeros(len(summary))
        # for A in range(0, len(summary),1):
        #     Radii[A] = summary[A][0]
        #     VPFs[A] = summary[A][1][0]

        allCiCprobs=np.array([])
        for A in range(0, len(summary),1):
            Radii[A] = summary[A][0]
            curr=np.array(resultsCiC[A][1][:])
            allCiCprobs = np.concatenate([allCiCprobs, curr])

        print(Radii, Radii.shape)
        print(' ')
        # print(VPFs, VPFs.shape)
        print(allCiCprobs, allCiCprobs.shape)
        # final=np.stack((Radii, VPFs), axis=1)
        # final=np.hstack((Radii, allCiCprobs))

        # LAP: so allCiCprobs holds probabilities, corresponding to: what radius (the row) and what number of galaxies (the column); total columns are numpN
        # LAP: should final be a dictionary instead? with radius = Radii, then CiC = allCiCprobs?

        print(final, final.shape)

        return xr.DataArray(
            final,
            dims=('r', 'CiC'),
            coords = {
                'r': Radii,
                # 'P0s': VPFs,
                'CiC': allCiCprobs,
            },
        )
