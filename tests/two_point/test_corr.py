import numpy as np
from summarizer.two_point import TPCF


def test__store(dummy_catalogue):
    r_bins = np.linspace(0.01, 0.2, 10)
    tpcf_summarizer = TPCF(r_bins=r_bins)
    tpcf = tpcf_summarizer(dummy_catalogue)
    tpcf_summarizer.store_summary("test.npy", tpcf)
    recovered_summary = np.load("test.npy")
    np.testing.assert_array_equal(tpcf, recovered_summary)
