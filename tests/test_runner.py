import numpy as np
from pathlib import Path
from summarizer.data import Catalogue
from summarizer.two_point import TPCF
from summarizer.runner import SummaryRunner


def test_config_summarizer():
    runner = SummaryRunner.from_config()
    assert isinstance(runner.summarizer, TPCF)
    np.testing.assert_equal(runner.summarizer.r_bins, [0.1, 1.0, 2.0])
    assert runner.summarizer.n_threads == 1


def test_config_catalogues():
    nodes = [
        0,
        2,
    ]
    catalogues = []
    for node in nodes:
        catalogues.append(
            Catalogue.from_quijote(
                node=node,
                redshift=0.5,
                path_to_lhcs=Path(
                    "/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/quijote/latin_hypercube"
                ),
                n_halos=100,
            )
        )
    runner = SummaryRunner.from_config()
    runner_names = [cat.name for cat in runner.catalogues]
    names = [cat.name for cat in catalogues]
    np.testing.assert_equal(names, runner_names)

    runner_length = [len(cat) for cat in runner.catalogues]
    length = [len(cat) for cat in catalogues]
    np.testing.assert_equal(runner_length, length)
