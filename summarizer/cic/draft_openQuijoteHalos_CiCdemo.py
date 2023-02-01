import numpy as np
from summarizer.data import Catalogue
from summarizer.two_point import corr
# from summarizer.vpf import vpf
from summarizer.cic import cic
import xarray as xr


'''
things that would be good to do:
- have choice between redshift space and config/real space from quijote
'''

#catalogue = Catalogue.from_quijote()
seed=10 #must be 1999 or less, only 2000 of these exist
quijote_path='/mnt/ceph/users/fvillaescusa/Quijote/Halos/FoF/latin_hypercube/HR_%d'%seed
print(quijote_path)
catalogue = Catalogue.from_quijote(node=0, redshift=0., n_halos=2000, path_to_lhcs=quijote_path)
print('Catalogue shape if ask for 2000 halos, fresh: ',catalogue.pos.shape)
real_Xs=catalogue.pos[:,0]
real_Ys=catalogue.pos[:,1]
real_Zs=catalogue.pos[:,2]
print('real Xs?', real_Xs)
print(np.amax(real_Xs), np.amax(real_Ys), np.amax(real_Zs))
# XYZ=catalogue.pos.T
# print(XYZ.shape)

from Corrfunc.theory.vpf import vpf
raw_VPF_calc=vpf(rmax=100, nbins=3, nspheres=10000, numpN=5, seed=4, X=real_Xs, Y=real_Ys, Z=real_Zs, max_cells_per_dim=100, periodic=True, boxsize=1000)
print(raw_VPF_calc, ' if try raw vpf calc without all the build up')

# real_VPF = vpf.VPF(100, 3, 10000, 1000)
real_CiC = cic.CiC(100, 3, 10000, 1000, 5)
# print(real_VPF)
results=real_CiC(catalogue)
print(results, results.shape)
print(results[0], len(results[0]))
print(real_CiC.to_dataset(results))
CiC.store_summary('testCiCruns.txt', results)
CiCchecks = xr.open_dataarray('testCiCruns.txt')
print(CiCchecks)
#ok, so each element corresponds to a bin number, holds a small array with [R, VPF]


# from Corrfunc.theory.vpf import vpf
# raw_VPF_calc=vpf(rmax=100, nbins=3, nspheres=10000, numpN=5, seed=4, X=real_Xs, Y=real_Ys, Z=real_Zs, max_cells_per_dim=100, periodic=True, boxsize=1000)
# print(raw_VPF_calc, ' if try raw vpf calc without all the build up')
# '''ok so it looks like the VPF calc fails for some reason if you give it numpN=1, need higher value'''
#
# print(' ')
# print('now trying the 2ptCF...')
# XYZ=np.random.rand(2000,3)
# boxsize=1000.
# XYZ=XYZ*boxsize
# print(np.amax(XYZ[:,0]), np.amax(XYZ[:,1]), np.amax(XYZ[:,2]))
#
# r_bins = np.arange(10,100,10)
# mu_bins = [-0.5,0.5]
# ells = [0,2,4]
# tpcf = corr.TPCF(r_bins, mu_bins, ells)
# tpcf(catalogue)
# print(tpcf(catalogue))
