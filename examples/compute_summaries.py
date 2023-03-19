from summarizer.runner import SummaryRunner
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--config_path', type=str)
args = parser.parse_args()

runner = SummaryRunner.from_config(args.config_path)
runner()
