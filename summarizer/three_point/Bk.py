import numpy as np
import xarray as xr
from typing import List
from summarizer.data import Catalogue
from summarizer.base import BaseSummary
import sys, os
import summarizer.three_point.bskit_main as bskit


class Bk(BaseSummary):
    def __init__(
        self,
        n_grid: int = 360,
        BoxSize: float = 1000,
        kmin: float = 0.01,
        kmax: float = 1.0,
        dk: float = 0.05,
        num_low_k_bins: int = 0,
        dk_high: float = -1.0,
        squeezed_bin_index: int = 0,
        isos_mult: float = 0.0,
        isos_tol: float = 0.1,
        for_grid_info_only: bool = False,
        pos_fac: float = 1.0,
        meas_type: str = "full",
        triangle_type: str = "equilateral",
    ):
        """Compute two point power spectrum (in fourier space),
        using Nbkit

        Args:
            n_grid (int): compute the density field on a regular grid with grid x grid x grid voxels
            BoxSize : float, optional
                the size of the box
            dk : float, optional
                the linear spacing of k bins to use; if not provided, the
                fundamental mode of the box is used
            kmin : float, optional
                the lower edge of the first k bin to use. only optional if k_edges is specified
            kmax : float, optional
                the upper edge of the last k bin to use. only optional if k_edges is specified
            num_lowk_bins : float, optional
                number of low-k bins with width `dk`
            dk_high : float, optional
                width for higher-k bins
            pos_fac : float, optional
                conversion factor between position units of input grid and Mpc/h
            triangle_type : {'all', 'equilateral', 'isosceles', 'squeezed'}
                type of triangle to measure bispectrum on
            isos_mult : float, optional
                multiplier defining triangles
            isos_tol : float, optional
                fractioanl tolerance for the isosceles condition to be satisfied
            squeezed_bin_index : int, optional
                k bin index of k_long bin
            for_grid_info_only : bool, optional
                if true, don't paint input mesh(es) - saves time if only grid info is wanted
            meas_type : {'full', 'grid_info', 'unnorm_b_value'}
                The meas_type flag controls what information is computed and output:
                    - 'full': bispectrum values, triangle counts in bins, and k_means in bins
                    - 'grid_info': triangle counts in bins and k_means in bins
                    - 'unnorm_b_value': unnormalized bispectrum values (not divided by N_triangles in bin)
        """
        self.n_grid = n_grid
        self.kmin = kmin
        self.kmax = kmax
        self.dk = dk
        self.BoxSize = BoxSize
        self.squeezed_bin_index = squeezed_bin_index
        self.num_low_k_bins = num_low_k_bins
        self.dk_high = dk_high
        self.squeezed_bin_index = squeezed_bin_index
        self.isos_mult = isos_mult
        self.isos_tol = isos_tol
        self.for_grid_info_only = for_grid_info_only
        self.pos_fac = pos_fac
        self.meas_type = meas_type
        self.triangle_type = triangle_type

    def __str__(
        self,
    ):
        return "bk"

    def __call__(
        self,
        catalogue: Catalogue,
    ) -> np.array:
        """Given a catalogue, compute its bispectrum

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

        fftb = bskit.FFTBispectrum(
            mesh,
            Nmesh=self.n_grid,
            BoxSize=np.ones(3) * self.BoxSize,
            dk=self.dk,
            kmin=self.kmin,
            kmax=self.kmax,
            pos_units_mpcoverh=self.pos_fac,
            second=None,
            third=None,
            num_lowk_bins=self.num_low_k_bins,
            dk_high=self.dk_high,
            triangle_type=self.triangle_type,
            squeezed_bin_index=self.squeezed_bin_index,
            isos_mult=self.isos_mult,
            isos_tol=self.isos_tol,
            for_grid_info_only=self.for_grid_info_only,
        )
        num_k_bins = len(
            bskit.generate_bin_edge_list(
                fftb.attrs["kmin"],
                fftb.attrs["kmax"],
                fftb.attrs["dk"],
                fftb.attrs["num_lowk_bins"],
                fftb.attrs["dk_high"],
            )
        )
        start_i = 0
        end_i = num_k_bins
        fftb.measure_bispectrum(
            imin=start_i, imax=end_i, verbose=0, meas_type=self.meas_type
        )
        Bk_obj_fftb = fftb.b
        return np.array(
            [
                Bk_obj_fftb["k_mean"][:, 0],
                Bk_obj_fftb["k_mean"][:, 1],
                Bk_obj_fftb["k_mean"][:, 2],
                Bk_obj_fftb["B"],
            ]
        ).T

    def to_dataset(self, summary: np.array) -> xr.DataArray:
        """Convert a power spectrum array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        return xr.DataArray(
            summary[:, -1],
            dims=("k"),
            coords={
                "k": summary[:, 0],
            },
        )
