import numpy as np
import xarray as xr
from typing import List, Union
import MAS_library as MASL
import Pk_library as PKL
import smoothing_library as SL
from summarizer.data import Catalogue
from summarizer.base import BaseSummary
import ast

class Bk(BaseSummary):
    def __init__(
        self,
        grid: int,
        BoxSize: float,
        k1: Union[List[float], np.array],
        k2: Union[List[float], np.array],
        theta: Union[List[float], np.array],
        reduced_bispectrum: bool = False,
        MAS: str = "CIC",
        smooth_field : bool = False,
        R_smooth : float = 0.0,
        Filter : str = "Top-Hat",
        n_threads: int = 1,
    ):
        """Compute three point bispectrum (in fourier space),
        using Pylians
        Args:
            grid (int): compute the density field on a regular grid with grid x grid x grid voxels
            BoxSize (float): Size of the periodic box. The units of the output power spectrum depend on this.
            k1 (Union[List[float], np.array]): array of one leg of the triangle
            k2 (Union[List[float], np.array]): array of the other leg of the triangle
            theta (Union[List[float], np.array]): array of the angle between the two legs of the triangle
            reduced_bispectrum (bool, optional): If True, compute the reduced bispectrum. Defaults to False.
            MAS (str, optional): Mass Assignment Scheme. Defaults to "CIC".
            smooth_field (bool):  If True, the field is smoothed with a filter having a smoothing scale R_smooth.
            R_smooth (float):  Smoothing scale of the filter.
            Filter (str):  Filter used to smooth the field. It can be "Top-Hat", "Gaussian" or "Gaussian-Window".
            n_threads (int, optional): number of threads for each tpcf. Defaults to 1.
        """
        self.grid = grid
        self.BoxSize = BoxSize
        self.MAS = MAS
        # print(k1, k2, theta)        
        self.k1 = np.array((k1))
        self.k2 = np.array((k2))

        # self.k1 = np.array(k1)
        # self.k2 = np.array(k2)
        self.theta = np.array((theta))
        if type(self.theta) is float:
            self.theta = np.array([theta])
        else:
            self.theta = theta
        
        self.reduced_bispectrum = reduced_bispectrum
        self.smooth_field = smooth_field
        self.R_smooth = R_smooth
        self.Filter = Filter
        self.n_threads = n_threads

    def __str__(self,):
        return 'Bk'

    def __call__(self, catalogue: Catalogue, dtype=np.float32) -> np.array:
        """ Given a catalogue, compute its Bispectrum
        Args:
            catalogue (Catalogue):  catalogue to summarize
        Returns:
            np.array: 
        """

        delta = np.zeros((self.grid,self.grid,self.grid), dtype=dtype)
        pos = (catalogue.pos).astype(dtype)
        MASL.MA(pos,delta,self.BoxSize,self.MAS)
        delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

        if self.smooth_field:
            W_k = SL.FT_filter(self.BoxSize, self.R_smooth, self.Filter, self.n_threads)
            delta = SL.field_smoothing(delta, W_k, self.n_threads)

        Bk_all = np.zeros((len(self.k1),len(self.k2), len(self.theta)), dtype=np.float64)
        Qk_all = np.zeros((len(self.k1),len(self.k2), len(self.theta)), dtype=np.float64)        
        for i1 in range(len(self.k1)):
            for i2 in range(len(self.k2)):
                if i2 >= i1:
                    BBk = PKL.Bk(delta, self.BoxSize, self.k1[i1], self.k2[i2], np.array(self.theta), self.MAS, self.n_threads)
                    Bk  = BBk.B     #bispectrum
                    Qk  = BBk.Q     #reduced bispectrum
                    Bk_all[i1,i2,:] = Bk
                    Qk_all[i1,i2,:] = Qk
                else:
                    Bk_all[i1,i2,:] = Bk_all[i2,i1,:]
                    Qk_all[i1,i2,:] = Qk_all[i2,i1,:]


        return np.array([Bk_all, Qk_all])


    def to_dataset(self, summary: np.array)->xr.DataArray:
        """ Convert a Bispectrum into an xarray dataset
        with coordinates
        Args:
            summary (np.array): summary to convert
        Returns:
            xr.DataArray: dataset array
        """
        if self.reduced_bispectrum:
            summary_tosave = summary[1]
        else:
            summary_tosave = summary[0]

        return xr.DataArray(
            summary_tosave,
            coords={
                "k1": self.k1,
                "k2": self.k2,
                "theta": self.theta,
            },
            dims=["k1", "k2", "theta"],
        )