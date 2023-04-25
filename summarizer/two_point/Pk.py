import numpy as np
import xarray as xr
from typing import List 
from summarizer.data import Catalogue
from summarizer.base import BaseSummary
from nbodykit.algorithms.fftpower import FFTPower


class Pk(BaseSummary):
    def __init__(
        self,
        n_grid: int = 360,
        dk: float = 0.005,
        ells: List[int] = [0, 2, 4],
        LOS: List[int] = [0, 0, 1],
    ):
        """Compute two point power spectrum (in fourier space),
        using nbodykit

        Args:
            n_grid (int): compute the density field on a regular grid with grid x grid x grid voxels
            dk (float): spacing in k bins.
            ells (List[int]): list of multipoles to compute
            LOS (List[int]): line of sight (x,y,z)
        """
        self.n_grid = n_grid
        self.dk = dk
        self.ells = ells
        self.LOS = LOS

    def __str__(self,):
        return 'pk'

    def __call__(
        self,
        catalogue: Catalogue,
    ) -> np.array:
        """Given a catalogue, compute its power spectrum

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array:
        """
        if hasattr(catalogue, "mesh"):
            assert catalogue.mesh.preview().shape == (
                self.n_grid,
                self.n_grid,
                self.n_grid,
            ), "Mesh has wrong shape!"
            mesh = catalogue.mesh
        else:
            mesh = catalogue.to_mesh(
                n_mesh=self.n_grid,
                resampler="tsc",
            )
        pk_moments = FFTPower(
            mesh,
            mode="2d",
            dk=self.dk,
            kmin=0.0,
            poles=self.ells,
            los=self.LOS,
        )
        k = pk_moments.poles["k"]
        pks = []
        for ell in self.ells:
            multipole = pk_moments.poles[f"power_{ell}"].real
            if ell == 0:
                multipole -= pk_moments.attrs["shotnoise"]
            pks.append(multipole)
        pks.append(k)
        return np.vstack(pks)

    def to_dataset(self, summary: np.array) -> xr.DataArray:
        """Convert a power spectrum array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        return xr.DataArray(
            summary[:-1],
            dims=("ells", "k"),
            coords={
                "ells": self.ells,
                "k": summary[-1],
            },
        )
