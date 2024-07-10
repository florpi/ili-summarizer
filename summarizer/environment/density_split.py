import numpy as np
import xarray as xr
from typing import List, Union
from pycorr import TwoPointCorrelationFunction
from pandas import qcut

from summarizer.data import BoxCatalogue, SurveyCatalogue
from summarizer.base import BaseSummary
from summarizer.utils import compute_overdensity

class DensitySplit(BaseSummary):
    def __init__(
        self,
        r_bins: Union[str, List],
        mu_bins: Union[str, List],
        ells: List[int],
        n_quantiles: int = 5,
        smoothing_radius: float = 15.0,
        n_threads: int = 1,
    ):
        """Compute density split statistics for a catalogue of tracers.
        See arXiv:2209.04310 for more on the particular summary statistic used

        Args:
            r_bins (np.array): bins in pair separation
            n_quantiles (int, optional): Number of quantiles to split the sample into. Defaults to 5.
            smoothing_radius (float, optional): smoothing radius used to compute density around random points.
            Defaults to 20.0.
            n_threads (int, optional): number of threads to use. Defaults to 1.
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
        self.n_quantiles = n_quantiles
        self.smoothing_radius = smoothing_radius
        self.n_threads = n_threads

    def __str__(self,):
        return 'density_split'

    def get_quantiles(self, sampling_positions: np.array, density: np.array, n_quantiles: int) -> List[np.array]:
        """ Split the sampling positions according to their local density into
        a number of quantiles

        Args:
            sampling_positions (np.array): random positions 
            density (np.array): density around random positions 
            n_quantiles (int): number of quantiles to split the sample into 

        Returns:
            List[np.array]: list of random positions in each quantile 
        """
        quantiles_idx = qcut(density, n_quantiles, labels=False)
        return [
            sampling_positions[quantiles_idx == i] for i in range(n_quantiles)
        ]

    def __call__(
        self, 
        catalogue: Union[SurveyCatalogue, BoxCatalogue],
        return_dataset: bool = False,
    ) -> np.array:
        """Given a catalogue, compute the density split statistics

        Args:
            catalogue (Catalogue): catalogue to summarize

        Returns:
            np.array: density split statistics
        """
        if catalogue.is_periodic_box:
            random_points = np.random.uniform(
                0, catalogue.boxsize, (self.n_quantiles * len(catalogue), 3)
            )
        else:
            random_points = catalogue.randoms_pos
        overdensity = compute_overdensity(
            query_positions=random_points,
            catalogue=catalogue,
            smoothing_radius=self.smoothing_radius,
        )
        quantiles = self.get_quantiles(
            sampling_positions=random_points,
            density=overdensity,
            n_quantiles=self.n_quantiles,
        )
        cross_correlations = []
        for i in range(self.n_quantiles):
            if catalogue.is_periodic_box:
                result = TwoPointCorrelationFunction(
                    "smu",
                    edges=(self.r_bins, self.mu_bins),
                    data_positions1=quantiles[i],
                    data_positions2=catalogue.galaxies_pos,
                    data_weights1=catalogue.galaxies_weights,
                    engine="corrfunc",
                    n_threads=self.n_threads,
                    compute_sepsavg=False,
                    position_type='pos',
                    boxsize=catalogue.boxsize,
                    los='z',
                )(ells=self.ells)
            else:
                #TODO: Make sure we are doing the right thing with quantiles randoms
                result = TwoPointCorrelationFunction(
                    "smu",
                    edges=(self.r_bins, self.mu_bins),
                    data_positions1=quantiles[i],
                    data_positions2=catalogue.galaxies_pos,
                    data_weights1=catalogue.galaxies_weights,
                    randoms_positions1=catalogue.randoms_pos,
                    randoms_positions2=catalogue.randoms_pos,
                    engine="corrfunc",
                    n_threads=self.n_threads,
                    boxsize=catalogue.boxsize,
                    los='z',
                )(ells=self.ells)
            cross_correlations.append(result)
        if return_dataset:
            return self.to_dataset(np.array(cross_correlations))
        return np.array(cross_correlations)

    def to_dataset(self, summary: np.array)->xr.DataArray:
        """ Convert a density split array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        return xr.DataArray(
            summary,
            dims=('quantile', 'ells','r'),
            coords = {
                'quantile': list(range(self.n_quantiles)),
                'ells': self.ells,
                'r': self.r,
            },
        )

