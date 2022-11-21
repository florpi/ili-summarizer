import yaml
import time
import logging
import importlib
from typing import Optional, List, Dict
from pathlib import Path
from mpi4py import MPI
from summarizer.data import Catalogue
from summarizer.base import BaseSummary

logging.basicConfig(level = logging.INFO)

default_config = Path(__file__).parent.parent / "examples/configs/sample_config.yaml"
#TODO: When MPI, each rank should only read the catalogues it needs (reading slow)

class SummaryRunner:
    def __init__(
        self,
        summarizer: BaseSummary,
        catalogues: List[Catalogue],
        output_path: Path,
    ):
        """Class to generate summaries from arrays of simulated data

        Args:
            summarizer (BaseSummary): summary to use
            catalogues (List[Catalogue]): list of data catalogues
            output_path (Path): path where to store outputs
        """
        self.summarizer = summarizer
        self.catalogues = catalogues
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        config_path: Path = default_config,
    )->"SummaryRunner":
        """Create a summary runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file. Defaults to default_config.

        Returns:
            SummaryRunner: the summary runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)
        summarizer = cls.load_summarizer(config["summarizer"])
        catalogues = cls.load_catalogues(config["catalogues"])
        output_path = Path(config['output_path'])
        redshift = config['catalogues']['args']['redshift']
        output_path = output_path / f"z_{redshift:.2f}"
        return cls(
            summarizer=summarizer,
            catalogues=catalogues,
            output_path=output_path,
        )

    @classmethod
    def load_summarizer(cls, summarizer_config: Dict)->BaseSummary:
        """Load the right summarizer, according to config file

        Args:
            summarizer_config (Dict): dictionary with the configuration for the summarizer

        Returns:
            summarizer (BaseSummary): the summarizer of choice
        """
        module = importlib.import_module(summarizer_config["module"])
        return getattr(
            module,
            summarizer_config["class"],
        )(**summarizer_config["args"])

    @classmethod
    def load_catalogues(cls, catalogues_config: Dict)->List[Catalogue]:
        """load simulated catalogues from config file

        Args:
            catalogues_config (Dict): dictionary with the configuration for the summarizer 

        Returns:
            catalogues (List[Catalogue]): list of catalogues
        """
        constructor = getattr(
            Catalogue, f'from_{catalogues_config["simulation_suite"]}'
        )
        catalogues = []
        if type(catalogues_config['nodes']) is str:
            nodes = eval(catalogues_config['nodes'])
        else:
            nodes = catalogues_config['nodes']
        for node in nodes:
            catalogues.append(constructor(node=node, **catalogues_config["args"]))
        return catalogues

    def __call__(
        self, 
    ):
        """Generate the summaries and store them to file
        """
        t0 = time.time()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        n_sims_per_core = len(self.catalogues) // size
        if rank == size - 1:
            catalogues = self.catalogues[rank * n_sims_per_core :]
        else:
            catalogues = self.catalogues[
                rank * n_sims_per_core: (rank + 1) * n_sims_per_core 
            ]
        for catalogue in catalogues:
            summary = self.summarizer(catalogue)
            self.summarizer.store_summary(
                self.output_path / f"{str(catalogue)}.nc", summary
            )
        logging.info(f'It took {time.time() - t0} seconds to compute all summaries')
