
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from summarizer.data import BoxCatalogue, SurveyCatalogue 
from summarizer.two_point import TwoPCF 
from cmass.summaries.tools import get_nofz
import nbodykit.lab as nblab


cat = BoxCatalogue.from_quijote(
    node=0,
    redshift=0.5,
    n_halos=10_000,
    path_to_lhcs= Path('/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/quijote/latin_hypercube'),
)
r_bins = np.logspace(-1, np.log10(150.), 50)
mu_bins = np.linspace(-1, 1, 120)
tpcf_summarizer = TwoPCF(r_bins=r_bins, mu_bins=mu_bins,ells=[0,2])
tpcf = tpcf_summarizer(cat)
tpcf = tpcf_summarizer.to_dataset(tpcf)


galaxies_ra_dec_z = nblab.transform.CartesianToSky(cat.galaxies_pos, cosmo=nblab.cosmology.Planck15).compute().T

randoms_pos = np.random.uniform(0, cat.boxsize, size=(10*len(cat), 3))
randoms_ra_dec_z = nblab.transform.CartesianToSky(randoms_pos, cosmo=nblab.cosmology.Planck15).compute().T


fiducial_cosmology= nblab.cosmology.Planck15
fsky = 0.8 
ng_of_z = get_nofz(galaxies_ra_dec_z[:, -1], fsky, cosmo=fiducial_cosmology)
galaxies_nbar = ng_of_z(galaxies_ra_dec_z[:, -1])
randoms_nbar = ng_of_z(randoms_ra_dec_z[:, -1])


survey_cat = SurveyCatalogue(
    galaxies_ra_dec_z = galaxies_ra_dec_z,
    randoms_ra_dec_z= randoms_ra_dec_z,
    redshift = 0.5,
    galaxies_nbar = galaxies_nbar,
    randoms_nbar = randoms_nbar, 
    fiducial_cosmology= fiducial_cosmology,
)


tpcf_survey = tpcf_summarizer(survey_cat)
tpcf_survey = tpcf_summarizer.to_dataset(tpcf_survey)

plt.loglog(tpcf.r, tpcf.sel(ells=0).values,label='Box')
plt.loglog(tpcf_survey.r, tpcf_survey.sel(ells=0).values,label='Survey')
plt.legend()
plt.xlabel('r [Mpc/h]')
plt.ylabel('TPCF')
plt.savefig('box_survey.png')