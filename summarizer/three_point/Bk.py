import numpy as np
import xarray as xr
from typing import List, Union, Optional
from summarizer.data import BoxCatalogue, SurveyCatalogue
from summarizer.base import BaseSummary
import sys, os
import PolyBin3D as pb


class Bk(BaseSummary):
    def __init__(
        self,
        k_bins: Union[str, List],
        n_mesh: int = 360,
        lmax: int = 2,
        mask = None,
        los:str = 'z',
        ells: List[int] = (0,2),
    ):
        """
        """
        self.k_bins = k_bins
        self.n_mesh = n_mesh 
        self.lmax = lmax
        self.mask = mask
        self.ells = ells
        self.los = los



    def __str__(
        self,
    ):
        return "bk"

    def __call__(
        self,
        catalogue: Union[BoxCatalogue, SurveyCatalogue],
    ) -> np.array:
        """Given a catalogue, compute its bispectrum

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array:
        """
        galaxies_mesh = catalogue.get_mesh(self.n_mesh)
        base = pb.PolyBin3D(
            sightline=self.los,
            gridsize=self.n_mesh,
            boxsize=catalogue.boxsize,
            boxcenter=0.5*catalogue.boxsize if catalogue.boxsize is not None else None,
            pixel_window='interlaced-tsc' if not catalogue.is_periodic_box else None,
        )
        bspec = pb.BSpec(
            base, 
            self.k_bins, # k-bin edges
            lmax=self.lmax, # Legendre multipoles
            mask=self.mask, # real-space mask
            applySinv=None, # filter to apply to data
            k_bins_squeeze = None, 
            include_partial_triangles=False,
        )
        bk_corr = bspec.Bk_ideal(
            mesh, 
            discreteness_correction=True,
        )
        k123 = bspec.get_ks()
        weight = k123.prod(axis=0)
        return np.array(
            [bk_corr[f'b{ell}'] * weight for ell in self.ells]
        )


    def to_dataset(self, summary: np.array) -> xr.DataArray:
        """Convert a power spectrum array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        return xr.DataArray(
            summary,
            dims=("k"),
            coords={
                "k": self.k_bins,
            },
        )
