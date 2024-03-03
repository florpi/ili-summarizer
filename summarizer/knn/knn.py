import xarray as xr
import numpy as np
from typing import List, Union
from summarizer.knn.compute_knn import get_CDFkNN
from summarizer.data import Catalogue
from summarizer.base import BaseSummary

class KNN(BaseSummary):
    def __init__(
        self,
        r_bins: Union[str, List],
        k: Union[str, List],
        n_threads: int = 1,
    ):
        """ Compute the knn summary statistic (https://arxiv.org/abs/2007.13342)

        Args:
            r_bins (Union[str,List]): bins in pair separation
            k (Union[str, List]): list of number of neighbors
            n_threads (int, optional): number of threads for each tpcf. Defaults to 1.
        """
        if type(r_bins) is str:
            self.r_bins = eval(r_bins)
        else:
            self.r_bins = np.array(r_bins)
        if type(k) is str:
            self.k = eval(k)
        else:
            self.k = np.array(k)
        self.n_threads = n_threads
    
    def __str__(self,):
        return 'knn'

    def __call__(self, catalogue: Catalogue) -> np.array:
        """ Given a catalogue, compute its knn statistics

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array: CDF of the volume determined by the k-nearest neighbors 
            at random, volume filling positions
        """
        random_pos = catalogue.boxsize*np.random.uniform(size=(len(catalogue), 3))
        return get_CDFkNN(
            r_bins=self.r_bins,
            pos = catalogue.galaxies_pos,
            random_pos = random_pos,
            boxsize=catalogue.boxsize,
            k=self.k,
            n_threads=self.n_threads,
        )

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
            dims=('k','r'),
            coords = {
                'k': self.k,
                'r': self.r_bins,
            },
        )

