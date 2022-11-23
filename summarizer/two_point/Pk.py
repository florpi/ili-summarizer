import numpy as np
import xarray as xr
from typing import List, Union
import MAS_library as MASL
import Pk_library as PKL
import smoothing_library as SL
from summarizer.data import Catalogue
from summarizer.base import BaseSummary


class PKL(BaseSummary):
    def __init__(
        self,
        grid: int,
        BoxSize: float,
        MAS: str = "CIC",
        axis : int = 0,
        mode : str = "3D",
        smooth_field : bool = False,
        R_smooth : float = 0.0,
        Filter : str = "Top-Hat",
        n_threads: int = 1,
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
        self.grid = grid
        self.BoxSize = BoxSize
        self.MAS = MAS
        self.axis = axis
        self.smooth_field = smooth_field
        self.R_smooth = R_smooth
        self.Filter = Filter
        self.n_threads = n_threads
        self.mode = mode

    def __call__(self, catalogue: Catalogue, dtype=np.float32) -> np.array:
        """ Given a catalogue, compute its power spectrum

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array: 
        """

        delta = np.zeros((self.grid,self.grid,self.grid), dtype=dtype)
        pos = (catalogue.pos.T).astype(dtype)
        MASL.MA(pos,delta,self.BoxSize,self.MAS)
        delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

        if self.smooth_field:
            W_k = SL.FT_filter(self.BoxSize, self.R_smooth, self.Filter, self.n_threads)
            delta = SL.field_smoothing(delta, W_k, self.n_threads)

        Pk = PKL.Pk(delta, self.BoxSize, self.axis, self.MAS, self.n_threads)

        if self.mode == "1D":
            k1D      = Pk.k1D
            Pk1D     = Pk.Pk1D
            return np.array([k1D, Pk1D])
        elif self.mode == "2D":
            kpar     = Pk.kpar
            kper     = Pk.kper
            Pk2D     = Pk.Pk2D
            return np.array([kpar, kper, Pk2D])
        elif self.mode == "3D":
            k       = Pk.k3D
            Pk0     = Pk.Pk[:,0] #monopole
            Pk2     = Pk.Pk[:,1] #quadrupole
            Pk4     = Pk.Pk[:,2] #hexadecapole
            return np.array([k, Pk0, Pk2, Pk4])
        else:
            raise ValueError(f"mode {self.mode} not recognized")


    def to_dataset(self, summary: np.array)->xr.DataArray:
        """ Convert a power spectrum array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        if self.mode == "1D":
            k1D = summary[0]
            Pk1D = summary[1]
            return xr.DataArray(Pk1D, coords=[k1D], dims=["k1D"])
        elif self.mode == "2D":
            kpar = summary[0]
            kper = summary[1]
            Pk2D = summary[2]
            return xr.DataArray(Pk2D, coords=[kpar, kper], dims=["kpar", "kper"])
        elif self.mode == "3D":
            k = summary[0]
            Pk0 = summary[1]
            Pk2 = summary[2]
            Pk4 = summary[3]
            return xr.DataArray(np.array([Pk0, Pk2, Pk4]), coords=[k], dims=["k", "multipoles"])
        else:
            raise ValueError(f"mode {self.mode} not recognized")

