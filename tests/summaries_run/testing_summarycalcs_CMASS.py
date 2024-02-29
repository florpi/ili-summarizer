import numpy as np
from summarizer.data import Catalogue
from summarizer import Pk, Mk, DensitySplit, CiC, Bk, WST
# from summarizer import TwoPCF, Pk, Mk, DensitySplit, CiC, Bk, WST
import hydra
import nbodykit


'''
WHAT TO RUN ON RUSTY BEFORE TRYING TO DO THIS CODE:

module load openmpi
module load lib/hdf5
module load python-mpi
module load gcc
source /mnt/home/lperez/python-cmass/bin/activate
pip install -e ".[backends]"
pip install -e .
python -m pytest tests/summaries_run/test_all_summaries_work_CMASS.py

'''

def catalogue():
    return Catalogue(
        pos=np.random.uniform(0, 1000, (500, 3)), 
        vel=np.random.uniform(0, 1000, (500, 3)), 
        mass= np.random.random(500),
        redshift=None,
        cosmo_dict=None,
        name='test',
        boxsize=1000.,
        n_mesh=64,
    )
# print(catalogue())

print(' -------- ')
'---------'


# @hydra.main(version_base=None, config_path="../conf", config_name="config") 
@hydra.main(version_base=None, config_path="/mnt/home/lperez/ltu-cmass/cmass/conf", config_name="config") 
#^ this is what gives cfg for later steps, using yamls!  fix paths 
#^ this is what is causing catalogue_CMASS to pretend to be a none type later
# this will aLWYS give none, will have to do the catalogue.cmass....() inside each function to work
def catalogue_CMASS(cfg):
    return Catalogue.CMASS_proc(
        node=0, #
        cfg=cfg)
print('is this none?', type(catalogue_CMASS()))
print(catalogue_CMASS())
#will it be mad if it doesn't get a velocity field? 
#what does it print?


# '---------'
# print('---------')

# # def test_run_twopcf(catalogue):
# #     tpcf_runner = TwoPCF(
# #         r_bins=np.linspace(0.01, 20., 10),
# #         mu_bins=np.linspace(-1.,1.,210),
# #         ells=[0,2,4],
# #     )
# #     tpcf = tpcf_runner(catalogue)
# #     tpcf = tpcf_runner.to_dataset(tpcf)
# #     assert tpcf.values.shape == (3, 9)

def test_run_pk(catalogue):
    pk_runner = Pk(
        ells=[0,2],
        dk=0.005,
        n_grid=64,    
    )
    pk = pk_runner(catalogue=catalogue)
    print('~*~*~*~*~*~**~')
    print('output of test_run_pk!:', pk)
    print('~*~*~*~*~*~**~')
    pk = pk_runner.to_dataset(pk)
    # assert pk.values.shape == (2, 40)

    # pk['k'], pk['power'].real

print('~*~*~*with quijote catalogue?~*~*~**~')
# quij_cat=catalogue()
test_run_pk(catalogue())
print('~*~*~*~*~*~**~')
print(' ')
print('~*~*~*with CMASS weird catalogue?~*~*~**~')

# @hydra.main(version_base=None, config_path="/mnt/home/lperez/ltu-cmass/cmass/conf", config_name="config") 
@hydra.main(version_base=None, config_path="/mnt/home/lperez/ltu-cmass/cmass/conf", config_name="config") 
def test_run_pk_CMASS(cfg):
    catalogue_CMASS=Catalogue.CMASS_proc(
        node=0, #
        cfg=cfg)
    # print('positions? ',catalogue_CMASS.pos[:,0])
    # print('~*~*~*~*~*~**~')
    pk_runner = Pk(
        ells=[0,2],
        dk=0.005,
        n_grid=360,    
        # n_grid=64,    
    )
    pk = pk_runner(catalogue=catalogue_CMASS)
    pk = pk_runner.to_dataset(pk)
    print('~*~*~*~*~*~**~')
    print('output of test_run_pk!:', pk)
    print('~*~*~*~*~*~**~')
    # assert pk.values.shape == (2, 40)
test_run_pk_CMASS()
# # def test_run_Mk(catalogue):
# #     mk_runner = Mk(ells=[0,2])
# #     mk = mk_runner(catalogue=catalogue)
# #     mk = mk_runner.to_dataset(mk)
# #     print('~*~*~*~*~*~**~')
# #     print('output of test_run_Mk!:', mk)
# #     print('~*~*~*~*~*~**~')
# #     assert mk.values.shape == (2, 226)

# # @hydra.main(version_base=None, config_path="/mnt/home/lperez/ltu-cmass/cmass/conf", config_name="config") 
# def test_run_Mk_CMASS(cfg):
#     catalogue_CMASS=Catalogue.CMASS_proc(
#         node=0, #
#         cfg=cfg)
#     mk_runner = Mk(ells=[0,2])
#     mk = mk_runner(catalogue=catalogue_CMASS)
#     mk = mk_runner.to_dataset(mk)
#     print('~*~*~*~*~*~**~')
#     print('output of test_run_Mk!:', mk)
#     print('~*~*~*~*~*~**~')
#     assert mk.values.shape == (2, 226)

# #FAILED tests/summaries_run/test_all_summaries_work_CMASS.py::test_run_density_split - KeyError: 'gpu'
# # def test_run_density_split(catalogue):
# #     ds_runner = DensitySplit(
# #         r_bins=np.linspace(0.01, 20., 10),
# #         mu_bins=np.linspace(-1.,1.,210),
# #         ells=[0,2,4], 
# #     )
# #     ds = ds_runner(catalogue=catalogue)
# #     ds = ds_runner.to_dataset(ds)
# #     assert ds.values.shape == (5, 3, 9)

# # def test_run_cic(catalogue):
# #     cic_runner = CiC(
# #         r_max=30.,
# #         n_bins = 15,
# #         n_spheres=1000,
# #         num_pN=10,
# #     )
# #     cic = cic_runner(catalogue=catalogue)
# #     cic = cic_runner.to_dataset(cic)
# #     print('~*~*~*~*~*~**~')
# #     print('output of test_run_cic!:', cic)
# #     print('output of test_run_cic[0]!:', cic[0])
# #     print('~*~*~*~*~*~**~')
# #     assert cic.values.shape == (15, 10)  

# # @hydra.main(version_base=None, config_path="/mnt/home/lperez/ltu-cmass/cmass/conf", config_name="config") 
# def test_run_cic_CMASS(cfg):
#     catalogue_CMASS=Catalogue.CMASS_proc(
#         node=0, #
#         cfg=cfg)
    
#     cic_runner = CiC(
#         r_max=30., #likely needs updating
#         n_bins = 15,
#         n_spheres=1000,
#         num_pN=10,
#     )
#     cic = cic_runner(catalogue=catalogue_CMASS)
#     cic = cic_runner.to_dataset(cic)
#     print('~*~*~*~*~*~**~')
#     print('output of test_run_cic!:', cic)
#     print('output of test_run_cic[0]!:', cic[0])
#     print('~*~*~*~*~*~**~')
#     assert cic.values.shape == (15, 10)  

# # def test_run_bk(catalogue):
# #     bk_runner = Bk(
# #         n_grid=64,
# #         BoxSize = catalogue.boxsize,
# #         kmin = 0.01,  
# #         kmax = 1.,
# #         dk = 0.05
# #     )
# #     bk = bk_runner(catalogue=catalogue)
# #     bk = bk_runner.to_dataset(bk)
# #     print('~*~*~*~*~*~**~')
# #     print('output of test_run_bk!:', bk)
# #     print('~*~*~*~*~*~**~')
# #     assert bk.values.shape == (19,)


# '''
# this wILL need updating, it expects a boxsize!!! does:
# fftb = bskit.FFTBispectrum(mesh,Nmesh=self.n_grid,BoxSize=np.ones(3)*self.BoxSize,
#                 dk=self.dk,kmin=self.kmin,kmax=self.kmax,pos_units_mpcoverh=self.pos_fac,
#                 second=None,third=None,
#                 num_lowk_bins=self.num_low_k_bins,dk_high=self.dk_high,
#                 triangle_type=self.triangle_type,squeezed_bin_index=self.squeezed_bin_index,
#                 isos_mult=self.isos_mult,isos_tol=self.isos_tol,
#                 for_grid_info_only=self.for_grid_info_only)
# '''
# # @hydra.main(version_base=None, config_path="/mnt/home/lperez/ltu-cmass/cmass/conf", config_name="config") 
# def test_run_bk_CMASS(cfg):
#     catalogue_CMASS=Catalogue.CMASS_proc(
#         node=0, #
#         cfg=cfg)
    
#     bk_runner = Bk(
#         n_grid=64,
#         # BoxSize = catalogue_CMASS.boxsize,
#         kmin = 0.01,  
#         kmax = 1.,
#         dk = 0.05
#     )
#     bk = bk_runner(catalogue=catalogue_CMASS)
#     bk = bk_runner.to_dataset(bk)
#     print('~*~*~*~*~*~**~')
#     print('output of test_run_bk!:', bk)
#     print('~*~*~*~*~*~**~')
#     assert bk.values.shape == (19,)

# #FAILED tests/summaries_run/test_all_summaries_work_CMASS.py::test_run_wst - RuntimeError: Tensor must be of spatial size (360, 360, 360)
# # def test_run_wst(catalogue):
# #     wst_runner = WST()
# #     wst = wst_runner(catalogue=catalogue)
# #     wst = wst_runner.to_dataset(wst)
# #     assert wst.values.shape == (76,)