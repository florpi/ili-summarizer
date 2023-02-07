from pathlib import Path
from summarizer.data import Catalogue
from summarizer.two_point import Mk

def test_mk_runs():
    cat = Catalogue.from_quijote(
        node=0,
        redshift=0.5,
        path_to_lhcs= Path('/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/quijote/latin_hypercube'),
    )
    mk_runner = Mk(ells=[0,2])
    mk = mk_runner(catalogue=cat)
    mk = mk_runner.to_dataset(mk)
    assert mk.values.shape == (2, 226)