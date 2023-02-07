import pytest
import numpy as np
from summarizer.data import Catalogue
from summarizer import TwoPCF, Pk, Mk, DensitySplit, CiC, Bk


@pytest.fixture
def catalogue():
    return Catalogue(
        pos=np.random.uniform(0, 1000, (500, 3)), 
        vel=None,
        redshift=None,
        cosmo_dict=None,
        name='test',
        boxsize=1000.
    )

def test_run_twopcf(catalogue):
    tpcf_runner = TwoPCF(
        r_bins=np.linspace(0.01, 20., 10),
        mu_bins=np.linspace(-1.,1.,210),
        ells=[0,2,4],
    )
    tpcf = tpcf_runner(catalogue)
    tpcf = tpcf_runner.to_dataset(tpcf)
    assert tpcf.values.shape == (3, 9)

def test_run_pk(catalogue):
    pk_runner = Pk(
        ells=[0,2],
        dk=0.005,
        n_grid=360,    
    )
    pk = pk_runner(catalogue=catalogue)
    pk = pk_runner.to_dataset(pk)
    assert pk.values.shape == (2, 226)

def test_run_Mk(catalogue):
    mk_runner = Mk(ells=[0,2])
    mk = mk_runner(catalogue=catalogue)
    mk = mk_runner.to_dataset(mk)
    assert mk.values.shape == (2, 226)

def test_run_density_split(catalogue):
    ds_runner = DensitySplit(
        r_bins=np.linspace(0.01, 20., 10),
        mu_bins=np.linspace(-1.,1.,210),
        ells=[0,2,4], 
    )
    ds = ds_runner(catalogue=catalogue)
    ds = ds_runner.to_dataset(ds)
    assert ds.values.shape == (5, 3, 9)

def test_run_cic(catalogue):
    cic_runner = CiC(
        r_max=30.,
        n_bins = 15,
        n_spheres=1000,
        num_pN=10,
    )
    cic = cic_runner(catalogue=catalogue)
    cic = cic_runner.to_dataset(cic)
    assert cic.values.shape == (15, 10)  


def test_run_bk(catalogue):
    bk_runner = Bk(
        grid=100,
        BoxSize = catalogue.boxsize,
        k1 = [0.1,0.5,0.7],
        k2 = [0.1,0.5,0.7],
        theta = [0.1,0.5,0.7],
    )
    bk = bk_runner(catalogue=catalogue)
    bk = bk_runner.to_dataset(bk)
    assert bk.values.shape == (3,3,3)