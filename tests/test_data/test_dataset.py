from summarizer.dataset import Dataset
from pathlib import Path


def test__load_summary():
    ds = Dataset(
        nodes=[0,2],
        path_to_data = Path('../../examples/test_data/'),
        root_file='TPCF_z_0.50_quijote',
        filters=None
    )
    summary = ds.load_summary(node=0)
    assert len(summary.values) == len(summary['r'])

def test__load_summary_with_filters():
    ds = Dataset(
        nodes=[0,2],
        path_to_data = Path('../../examples/test_data/'),
        root_file='TPCF_z_0.50_quijote',
        filters={'r': (1.,10.)}
    )
    summary = ds.load_summary(node=0)
    assert len(summary.values) == 1

