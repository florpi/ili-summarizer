from abc import ABC, abstractmethod
import xarray as xr
import numpy as np
from summarizer.data import Catalogue


class BaseSummary(ABC):
    @abstractmethod
    def __call__(self, catalogue: Catalogue) -> np.array:
        """Given a catalogue, produce a summary

        Args:
            catalogue (Catalogue): catalogue to summarize

        Returns:
            np.array: summary 
        """
        return

    @abstractmethod
    def to_dataset(self, summary: np.array)->xr.DataArray:
        """Convert numpy array into xarray dataset with coordinates and dimensions

        Args:
            summary (np.array): summary to store

        Returns:
            dataset array
        """
        pass

    def store_summary(self, filename: str, summary: np.array):
        """Store summary as xarray dataset

        Args:
            filename (str): where to store 
            summary (np.array): summary to store
        """
        ds = self.to_dataset(summary)
        ds.to_netcdf(filename)
