import numpy as np
import xarray as xr
from typing import List, Union
from densitysplit.pipeline import DensitySplit as DensitySplitSummary
from pycorr import TwoPointCorrelationFunction
from nbodykitlab.fitlers import Gaussian

from summarizer.data import Catalogue
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

    def __call__(self, catalogue: Catalogue) -> np.array:
        """Given a catalogue, compute the density split statistics

        Args:
            catalogue (Catalogue): catalogue to summarize

        Returns:
            np.array: density split statistics
        """
        ds = DensitySplitSummary(catalogue.pos, catalogue.boxsize,)
        random_points = np.random.uniform(
            0, catalogue.boxsize, (5 * len(catalogue.pos), 3)
        )
        ds.density = compute_overdensity(
            eval_positions = random_points,
            filter= Gaussian(self.smoothing_radius),
            tracer_mesh=catalogue.mesh
        )
        quantiles = ds.get_quantiles(
            nquantiles=self.n_quantiles,
            return_density=False,
        )
        cross_correlations = []
        for i in range(self.n_quantiles):
            result = TwoPointCorrelationFunction(
                "smu",
                edges=(self.r_bins, self.mu_bins),
                data_positions1=quantiles[i],
                data_positions2=catalogue.pos,
                engine="corrfunc",
                boxsize=catalogue.boxsize,
                nthreads=self.n_threads,
                compute_sepsavg=False,
                position_type="pos",
                los='z',
            )
            cross_correlations.append(result(ells=self.ells))
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

