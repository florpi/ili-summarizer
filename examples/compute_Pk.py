import sys, os
sys.path.append('/global/cfs/cdirs/lsst/www/shivamp/ili-summarizer')
from summarizer.runner import SummaryRunner

runner = SummaryRunner.from_config('configs/sample_config_Pk.yaml')
runner()