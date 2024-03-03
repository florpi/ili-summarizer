import numpy as np
import xarray as xr
from typing import List, Union
from pycorr import TwoPointCorrelationFunction
from summarizer.data import BoxCatalogue, SurveyCatalogue
from summarizer.base import BaseSummary


class TwoPCF(BaseSummary):
    def __init__(
        self,
        r_bins: Union[str, List],
        mu_bins: Union[str, List],
        ells: List[int],
        n_threads: int = 1,
    ):
        """Compute two point correlation functions (in configuration space),
        using pycorr

        Args:
            r_bins (np.array): bins in pair separation
            mu_bins (np.array): bins in angle to the line of sight 
            ells (list): multipoles to compute
            n_threads (int, optional): number of threads for each tpcf. Defaults to 1.
        """
        if type(r_bins) is str:
            self.r_bins = eval(r_bins)
        else:
            self.r_bins = np.array(r_bins)
        self.r = 0.5*(self.r_bins[1:] + self.r_bins[:-1])
        if type(mu_bins) is str:
            self.mu_bins = eval(mu_bins)
        else:
            self.mu_bins = np.array(mu_bins)
        self.mu = 0.5*(self.mu_bins[1:] + self.mu_bins[:-1])
        self.ells = ells
        self.n_threads = n_threads

    def __str__(self,):
        return 'twopcf'

    def __call__(self, catalogue: Union[SurveyCatalogue, BoxCatalogue], return_dataset: bool =False,) -> np.array:
        """ Given a catalogue, compute its two point correlation function

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array: two-point correlation function
        """
        if catalogue.is_periodic_box:
            tpcf = TwoPointCorrelationFunction(
                "smu",
                edges=(self.r_bins, self.mu_bins),
                data_positions1=catalogue.galaxies_pos,
                data_weights1=catalogue.weights,
                engine="corrfunc",
                n_threads=self.n_threads,
                boxsize=catalogue.boxsize,
                position_type='pos',
                los='z',
            )(ells=self.ells)
        else:
            tpcf = TwoPointCorrelationFunction(
                "smu",
                edges=(self.r_bins, self.mu_bins),
                data_positions1=catalogue.galaxies_pos,
                data_weights1=catalogue.weights,
                randoms_positions1=catalogue.randoms_pos,
                engine="corrfunc",
                n_threads=self.n_threads,
                boxsize=catalogue.boxsize,
                position_type='pos',
                los='z',
            )(ells=self.ells)

        if return_dataset:
            return self.to_dataset(tpcf)
        return tpcf

    def to_dataset(self, summary: np.array)->xr.DataArray:
        """ Convert a tpcf array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        return xr.DataArray(
            summary,
            dims=('ells','r'),
            coords = {
                'ells': self.ells,
                'r': self.r,
            },
        )

