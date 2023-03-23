import yaml
import time
import logging
import importlib
from typing import Optional, List, Dict, Callable
from pathlib import Path
from summarizer.data import Catalogue
from summarizer.base import BaseSummary

logging.basicConfig(level=logging.INFO)

default_config = Path(__file__).parent.parent / "examples/configs/sample_config.yaml"


class SummaryRunner:
    def __init__(
        self,
        summarizers: List[BaseSummary],
        catalogue_loader: Callable,
        output_path: Path,
        idx_to_load: List[int] = None,
        path_to_data: Optional[Path] = None,
        dont_overwrite: bool = True,
    ):
        """Class to generate summaries from arrays of simulated data

        Args:
            summarizer (BaseSummary): summary to use
            catalogues (List[Catalogue]): list of data catalogues
            output_path (Path): path where to store outputs
            idx_to_load (Optional[List[int]], optional): indices of catalogues to load,
            if None it will load all in path_to_data. Defaults to None.
            path_to_data (Optional[Path], optional): path to data. Defaults to None.
            dont_overwrite (bool, optional): if True, will not overwrite existing files.
        """
        self.summarizers = summarizers
        self.catalogue_loader = catalogue_loader
        if idx_to_load is None:
            self.idx_to_load = [
                int(str(run).split("/")[-1]) for run in path_to_data.iterdir() if run.is_dir()
            ]
        else:
            self.idx_to_load = idx_to_load
        logging.info(f"Loading idx {self.idx_to_load}")
        self.output_path = output_path
        for summarizer in self.summarizers:
            summary_path = self.output_path / summarizer.__str__()
            if summary_path is not None:
                summary_path.mkdir(parents=True, exist_ok=True)
        self.dont_overwrite = dont_overwrite

    @classmethod
    def from_config(
        cls,
        config_path: Path = default_config,
    ) -> "SummaryRunner":
        """Create a summary runner from a yaml config file

        Args:
            config_path (Path, optional): path to config file. Defaults to default_config.

        Returns:
            SummaryRunner: the summary runner specified by the config file
        """
        with open(config_path, "r") as fd:
            config = yaml.safe_load(fd)
        summarizers = cls.load_summarizers(config["summarizers"])
        catalogue_loader = cls.load_catalogue_loader(config["catalogues"])
        if "idx_to_load" in config["catalogues"]:
            idx_to_load = config["catalogues"]["idx_to_load"]
            if type(idx_to_load) is str:
                idx_to_load = eval(idx_to_load)
        else:
            idx_to_load = None
        output_path = Path(config["output_path"])
        redshift = config["catalogues"]["args"]["redshift"]
        output_path = output_path / f"z_{redshift:.2f}"
        return cls(
            summarizers=summarizers,
            catalogue_loader=catalogue_loader,
            idx_to_load=idx_to_load,
            output_path=output_path,
            path_to_data=Path(config["catalogues"]["args"]["path_to_lhcs"]),
        )

    @classmethod
    def load_summarizers(cls, summarizer_config: List[Dict]) -> List[BaseSummary]:
        """Load the right summarizers, according to config file

        Args:
            summarizer_config (List[Dict]): list of dictionaries with the configuration of each summarizer

        Returns:
            summarizers (List[BaseSummary]): the summarizers of choice
        """
        summarizers = []
        for config in summarizer_config:
            summarizers.append(cls.load_summarizer(config))
        return summarizers

    @classmethod
    def load_summarizer(cls, summarizer_config: Dict) -> BaseSummary:
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
    def load_catalogue_loader(cls, catalogues_config: Dict) -> List[Catalogue]:
        """load simulated catalogues from config file

        Args:
            catalogues_config (Dict): dictionary with the configuration for the summarizer

        Returns:
            Callable: function that loads the catalogues given an index
        """
        constructor = getattr(
            Catalogue, f'from_{catalogues_config["simulation_suite"]}'
        )
        return lambda idx: constructor(idx, **catalogues_config["args"])

    def __call__(
        self,
    ):
        """Generate the summaries and store them to file"""
        from mpi4py import MPI
        t0 = time.time()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        n_sims_per_core = len(self.idx_to_load) // size
        if rank == size - 1:
            idx_per_core = self.idx_to_load[rank * n_sims_per_core :]
        else:
            idx_per_core = self.idx_to_load[
                rank * n_sims_per_core : (rank + 1) * n_sims_per_core
            ]
        for idx in idx_per_core:
            catalogue = self.catalogue_loader(idx)
            for summarizer in self.summarizers:
                if self.dont_overwrite:
                    if (self.output_path / f"{str(summarizer)}/{str(catalogue)}.nc").exists():
                        logging.info(
                            f"Skipping {str(catalogue)} with {str(summarizer)} because it already exists"
                        )
                        continue
                summary = summarizer(catalogue)
                summarizer.store_summary(
                    self.output_path / f"{str(summarizer)}/{str(catalogue)}.nc", summary
                )
        logging.info(f"It took {time.time() - t0} seconds to compute all summaries")
