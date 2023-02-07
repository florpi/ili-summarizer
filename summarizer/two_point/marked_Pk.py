import numpy as np
import xarray as xr
from typing import List, Optional
from summarizer.data import Catalogue
from summarizer.base import BaseSummary
from summarizer.utils import compute_overdensity
from nbodykit.filters import TopHat
from nbodykit.algorithms.fftpower import FFTPower

class Mk(BaseSummary):
    def __init__(
        self,
        density_radius: Optional[float]=30.,
        mark_delta_s: Optional[float] = 0.1,
        mark_p: Optional[float] = 1.,
    ):
        self.density_radius = density_radius
        self.mark_delta_s = mark_delta_s
        self.mark_p = mark_p
    
    def get_mark(self, delta,)->float:
        return ((self.mark_delta_s+1.)/(self.mark_delta_s+ delta +1.0+1e-6))**self.mark_p

    def __call__(
        self,
        catalogue: Catalogue,
    )->np.array:

        delta_env = compute_overdensity(
            catalogue.pos,
            tracers_mesh = catalogue.mesh,
            filter=TopHat(r=self.density_radius),
        )
        marked_mesh = catalogue.to_mesh(
            n_mesh = self.n_grid,
            weights=self.get_mark(delta=delta_env),
        )
        pk_moments = FFTPower(
            marked_mesh,
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
