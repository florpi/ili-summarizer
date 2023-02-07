import numpy as np
import xarray as xr
from typing import Optional
from summarizer.data import Catalogue
from summarizer.base import BaseSummary
from pyspectrum import pyspectrum

class Bk(BaseSummary):
    def __init__(
        self,
        n_grid: int,
        n_max_steps: Optional[int] = 40,
        n_cut: Optional[int] = 3,
        step: Optional[float] = 3,
        fftw: Optional[str] ='pyfftw',
        n_threads: Optional[int] = 1,
    ):

        self.n_grid = n_grid
        self.n_max_steps = n_max_steps
        self.n_cut = n_cut
        self.step = step
        self.fftw = fftw
        self.n_threads = n_threads

    def __call__(self, catalogue: Catalogue,) -> np.array:
        """ Given a catalogue, compute its Bispectrum

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array: 
        """
        xyz = np.array(catalogue.pos).T
        bk = pyspectrum.Bk_periodic(
            xyz, 
            Lbox=catalogue.boxsize, 
            Ngrid=self.n_grid, 
            step=self.step, 
            Ncut=self.n_cut, 
            Nmax=self.n_max_steps, 
            fft=self.fft, 
            nthreads=self.n_threads, 
        )
        
        k1 = bk['meta']['kf'] * bk['i_k1']
        k2 = bk['meta']['kf'] * bk['i_k2']
        k3 = bk['meta']['kf'] * bk['i_k3']
        b123 = bk['b123']
        q123 = bk['q123']
        return np.vstack([k1, k2, k3, b123, q123])

    def to_dataset(self, summary: np.array)->xr.DataArray:
        """ Convert a Bispectrum into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        print(summary.shape)
        return xr.DataArray(
            summary[2:],
            coords={
                "k1": summary[0,],
                "k2": summary[1],
                "k3": summary[2],
                "reduced": (0,1),
            },
            dims=["reduced", "k1", "k2", "k3",],
        )


