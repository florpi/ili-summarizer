import numpy as np
from summarizer.data import Catalogue
from summarizer.two_point import corr
from summarizer.vpf import vpf
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

real_VPF = vpf.VPF(100, 3, 5000, 1000)
# print(real_VPF)
results=real_VPF(catalogue)
print(results, results.shape)
print(results[0], len(results[0]))
print('testing to_dataset: ', real_VPF.to_dataset(results))
real_VPF.store_summary('testVPFruns.txt', results)
VPFchecks = xr.open_dataarray('testVPFruns.txt')
print('did I load it correctly?', VPFchecks)
print(np.log10(VPFchecks[:]))
#ok, so each element corresponds to a bin number, holds a small array with [R, VPF]


from Corrfunc.theory.vpf import vpf
raw_VPF_calc=vpf(rmax=100, nbins=3, nspheres=5000, numpN=2, seed=4, X=real_Xs, Y=real_Ys, Z=real_Zs, max_cells_per_dim=100, periodic=True, boxsize=1000)
print('if try raw vpf calc without all the build up...')
# print(raw_VPF_calc)
p0_real = [t[1][0] for t in raw_VPF_calc]
print(np.log10(p0_real))

# '''ok so it looks like the VPF calc fails for some reason if you give it numpN=1, need higher value'''
# print(' ')
# print('let me test this for perfectly random points: ')

XYZ=np.random.rand(2000,3)
boxsize=1000.
XYZ=XYZ*boxsize
# print(np.amax(XYZ[:,0]), np.amax(XYZ[:,1]), np.amax(XYZ[:,2]))
raw_VPF_random=vpf(rmax=100, nbins=3, nspheres=15000, numpN=2, seed=4, X=XYZ[:,0], Y=XYZ[:,1], Z=XYZ[:,2], max_cells_per_dim=100, periodic=True, boxsize=1000)
print(raw_VPF_random)
density=2000./(boxsize**3.)
def Poisson_VPF_3d(density, radius):
    return (-1*density*(4./3.)*np.pi*(radius**3.))/(np.log(10.))

radii = [t[0] for t in raw_VPF_random]
p0_random = [t[1][0] for t in raw_VPF_random]
print(p0_random, np.log10(p0_random))
predp0= [Poisson_VPF_3d(density,radiii) for radiii in radii]
print(predp0)
# print(' ')
# print('now trying the 2ptCF...')
# r_bins = np.arange(10,100,10)
# mu_bins = [-0.5,0.5]
# ells = [0,2,4]
# tpcf = corr.TPCF(r_bins, mu_bins, ells)
# tpcf(catalogue)
# print(tpcf(catalogue))
