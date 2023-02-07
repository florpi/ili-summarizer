import numpy as np
import xarray as xr
from typing import List, Union
import MAS_library as MASL
import Pk_library as PKL
import smoothing_library as SL
from summarizer.data import Catalogue
from summarizer.base import BaseSummary
from nbodykit.algorithms.fftpower import FFTPower

class Pk(BaseSummary):
    def __init__(
        self,
        n_grid: int = 360,
        dk: float = 0.005,
        ells: List[int] = [0,2,4],
        LOS: List[int] = [0,0,1],
    ):
        """Compute two point power spectrum (in fourier space),
        using Pylians

        Args:
            grid (int): compute the density field on a regular grid with grid x grid x grid voxels
            BoxSize (float): Size of the periodic box. The units of the output power spectrum depend on this.
            MAS (str, optional): Mass Assignment Scheme. Defaults to "CIC".
            axis (int):  Axis along which compute the quadrupole, hexadecapole and the 2D power spectrum. If the field is in real-space set axis=0. If the field is in redshift-space set axis=0, axis=1 or axis=2 if the redshift-space distortions have been placed along the x-axis, y-axis or z-axis, respectively.
            mode (str): 1D, 2D or 3D. If mode=1D, the output power spectrum is integral over all perpendicular mode. If mode=2D, the output power spectrum will have parallel and perp k's. If mode=3D, the output power spectrum will have monopole, quadrupole and hexadecapole. You mostly want to have mode=3D and get the monopole which is the normal 3D power spectra
            smooth_field (bool):  If True, the field is smoothed with a filter having a smoothing scale R_smooth.
            R_smooth (float):  Smoothing scale of the filter.
            Filter (str):  Filter used to smooth the field. It can be "Top-Hat", "Gaussian" or "Gaussian-Window".
            n_threads (int, optional): number of threads for each tpcf. Defaults to 1.
        """
        self.n_grid = n_grid
        self.dk = dk
        self.ells = ells
        self.LOS = LOS


    def __call__(self, catalogue: Catalogue,) -> np.array:
        """ Given a catalogue, compute its power spectrum

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array: 
        """
        if hasattr(catalogue, 'mesh'):
            assert catalogue.mesh.preview().shape == (self.n_grid, self.n_grid, self.n_grid), 'Mesh has wrong shape!'
            mesh = catalogue.mesh
        else:
            mesh = catalogue.to_mesh(n_mesh=self.n_grid, resampler='tsc', )
        pk_moments = FFTPower(
            mesh,
            mode='2d',
            dk=self.dk,
            kmin=0.,
            poles=self.ells,
            los = self.LOS,
        )
        k = pk_moments.poles['k'] 
        pks = []
        for ell in self.ells:
            multipole = pk_moments.poles[f'power_{ell}'].real
            if ell == 0:
                multipole -= pk_moments.attrs['shotnoise']
            pks.append(multipole)
        pks.append(k)
        return np.vstack(pks)

    def to_dataset(self, summary: np.array)->xr.DataArray:
        """ Convert a power spectrum array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        return xr.DataArray(
            summary[:-1],
            dims=('ells', 'k'),
            coords={
                'ells': self.ells,
                'k': summary[-1],
            },
        )

