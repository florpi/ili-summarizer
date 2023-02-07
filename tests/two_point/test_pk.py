from pathlib import Path
from summarizer.data import Catalogue
from summarizer.two_point import Pk


def test_pk_runs():
    cat = Catalogue.from_quijote(
        node=0,
        redshift=0.5,
        path_to_lhcs= Path('/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/quijote/latin_hypercube'),
    )
    pk_runner = Pk(
        ells=[0,2],
        dk=0.005,
        n_grid=360,
    )
    pk = pk_runner(catalogue=cat)
    pk = pk_runner.to_dataset(pk)
    assert pk.values.shape == (2, 226)