import numpy as np
import xarray as xr
from typing import List 
from summarizer.data import Catalogue
from summarizer.base import BaseSummary
from nbodykit.algorithms.fftpower import FFTPower


class Pk(BaseSummary):
    def __init__(
        self,
        n_grid: int = 360,
        dk: float = 0.005,
        ells: List[int] = [0, 2, 4],
        LOS: List[int] = [0, 0, 1],
    ):
        """Compute two point power spectrum (in fourier space),
        using nbodykit

        Args:
            n_grid (int): compute the density field on a regular grid with grid x grid x grid voxels
            dk (float): spacing in k bins.
            ells (List[int]): list of multipoles to compute
            LOS (List[int]): line of sight (x,y,z)
        """
        self.n_grid = n_grid
        self.dk = dk
        self.ells = ells
        self.LOS = LOS

    def __str__(self,):
        return 'pk'

    def __call__(
        self,
        catalogue: Catalogue,
    ) -> np.array:
        """Given a catalogue, compute its power spectrum

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array:
        LAP replacing if hassattr bit with a try/except
        """
        # if hasattr(catalogue, "mesh"):
        #     assert catalogue.mesh.preview().shape == (
        #         self.n_grid,
        #         self.n_grid,
        #         self.n_grid,
        #     ), "Mesh has wrong shape!"
        #     mesh = catalogue.mesh
        # else:
        #     mesh = catalogue.to_mesh(
        #         n_mesh=self.n_grid,
        #         resampler="tsc",
        #     )
        try:
            print('trying catalogue.mesh')
            mesh = catalogue.mesh
            print('accessed catalogue.mesh directly')
            print(' ')
        except:
            try:
                print('trying catalogue.to_mesh(...)')
                mesh = catalogue.to_mesh(
                    n_mesh=self.n_grid,
                    resampler="tsc", #cic? 
                    compensated=True, #deaglan thinks this SHOULD be in!
                    )
                print('did catalogue.to_mesh()')
            except:
                return "Meshing is not going through! Recheck catalogue item"
        pk_moments = FFTPower(
            mesh,
            # mode="2d", #deaglan testing standard that is 1d here
            mode="1d", #deaglan testing standard that is 1d here
            dk=self.dk,
            kmin=0.0,
            # poles=self.ells, #these won't matter for 1D, deaglan thinks
            # los=self.LOS, #these won't matter for 1D, deaglan thinks
        )
        power =  pk_moments.power
        k = power["k"]
        pk = power['power'].real
        print('np.vstack((k, pk)): ',np.vstack((k, pk)).shape)
        return np.vstack((k, pk)) # this is where lucia is stuck!!!! to_dataset failing :(

        '''previous stuff for Carol's way of doing the 2d spectrum:'''
        # print('pk_moments? ', pk_moments)
        # k = pk_moments.poles["k"]
        # print('k?', k)
        # pks = []
        # for ell in self.ells:
        #     multipole = pk_moments.poles[f"power_{ell}"].real
        #     print('multipole for ', ell, ' is: ', multipole)
        #     if ell == 0:
        #         multipole -= pk_moments.attrs["shotnoise"]
        #     pks.append(multipole)
        # pks.append(k)
        # return np.vstack(pks)

    def to_dataset(self, summary: np.array) -> xr.DataArray:
        """Convert a power spectrum array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        # return xr.DataArray(
        #     summary[:-1],
        #     dims=("ells", "k"),
        #     coords={
        #         "ells": self.ells,
        #         "k": summary[-1],
        #     },
        # ) #   for 2d spectrum!!!!!
        return xr.DataArray(
            summary,
            dims=("k", "pk"),
            coords={
                "k": summary[0,:],
                "pk": summary[1,:],
            },
        )
#all the summaries use the mesh at some point! can we replace it with the 
#improved random mesh? wavelet and bispectrum do, at least
#not knn
    
#carol: have base catalog object that's a shared class that is rarely used
#then two separate box vs survey objects