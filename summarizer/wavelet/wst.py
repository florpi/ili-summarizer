import numpy as np
import xarray as xr
import torch
import time
import logging
from typing import List, Tuple 
from summarizer.data import Catalogue
from summarizer.base import BaseSummary
from kymatio.torch import HarmonicScattering3D
from kymatio.scattering3d.backend.torch_backend import TorchBackend3D


logging.basicConfig(level = logging.INFO)

class WST(BaseSummary):
    def __init__(
        self,
        J_3d: int = 4,
        L_3d: int = 4,
        integral_powers: List[float] = [
            0.8,
        ],
        sigma: float = 0.8,
        n_mesh: int = 360,
    ):
        """Compute the wavelet scattering transform coeffecients as in arxiv:2204.13717

        Args:
            J_3d (int, optional): number of dyadic scales. Defaults to 4.
            L_3d (int, optional): number of orientations. Defaults to 4.
            integral_powers (List[float], optional): powers to integrate over. Defaults to [0.8,].
            sigma (float, optional): width of the Gaussian window. Defaults to 0.8.
        """
        logging.info(f'Initializing wst')
        self.J_3d = J_3d
        self.L_3d = L_3d
        self.integral_powers = integral_powers
        self.sigma = sigma
        self.S = HarmonicScattering3D(
            J=self.J_3d,
            shape=(n_mesh, n_mesh, n_mesh,),
            L=self.L_3d,
            sigma_0=self.sigma,
            integral_powers=self.integral_powers,
            max_order=2,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.S.to(device)

    def __str__(
        self,
    ):
        return "wst"

    def __call__(self, catalogue: Catalogue) -> np.array:
        """Given a catalogue, compute its wst coefficients

        Args:
            catalogue (Catalogue):  catalogue to summarize

        Returns:
            np.array: wst coefficients
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f'Using device = {device}')
        density_field = catalogue.mesh.preview()
        full_density_batch = torch.from_numpy(np.asarray(np.real(density_field)))
        full_density_batch = full_density_batch.to(device).float()
        full_density_batch = full_density_batch.contiguous()
        w_orders_12 = self.S(full_density_batch)
        s_mat_avg = np.real(w_orders_12.cpu().numpy()[:, :, 0])
        s_mat_avg = s_mat_avg.flatten()
        test_shape = np.empty(
            [
                1,
                density_field.shape[0],
                density_field.shape[0],
                density_field.shape[0],
            ]
        )
        test_shape[0, :] = np.asarray(np.absolute(density_field))
        s0_batch = torch.from_numpy(test_shape)
        integr = TorchBackend3D.compute_integrals(s0_batch, self.integral_powers)
        s0 = integr.cpu().numpy()[0, 0]
        return np.hstack((s0, s_mat_avg))

    def to_dataset(self, summary: np.array) -> xr.DataArray:
        """Convert a wst array into an xarray dataset
        with coordinates

        Args:
            summary (np.array): summary to convert

        Returns:
            xr.DataArray: dataset array
        """
        return xr.DataArray(
            summary,
            dims=("coefficient",),
            coords={
                "coefficient": np.arange(summary.shape[0]),
            },
        )
