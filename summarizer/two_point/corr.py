import numpy as np
from pycorr import TwoPointCorrelationFunction
from summarizer.data import Catalogue
from summarizer.base import BaseSummary


class TPCF(BaseSummary):
    def __init__(
        self,
        r_bins: np.array,
        n_threads: int = 1,
    ):
        """Compute two point correlation functions (in configuration space),
        using pycorr

        Args:
            r_bins (np.array): bins in pair separation
            n_threads (int, optional): number of threads for each tpcf. Defaults to 1.
        """
        self.r_bins = r_bins
        self.n_threads = n_threads

    def __call__(self, catalogue: Catalogue) -> np.array:
        """ Given a catalogue, compute its two point correlation function

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array: two-point correlation function
        """
        return TwoPointCorrelationFunction(
            "s",
            edges=(self.r_bins,),
            data_positions1=catalogue.pos.T,
            engine="corrfunc",
            n_threads=self.n_threads,
            boxsize=catalogue.boxsize,
        ).corr

    def store_summary(self, filename: str, summary: np.array):
        """Store summary to numpy file

        Args:
            filename (str): where to store 
            summary (np.array): summary to store
        """
        np.save(
            filename,
            summary,
        )
