from pathlib import Path
from typing import List, Dict
import xarray as xr
import numpy as np

class Dataset:
    def __init__(
        self,
        nodes: List[int],
        path_to_data: Path,
        root_file: str, 
        filters: Dict=None,
    ):
        """Read dataset of summaries

        Args:
            nodes (List[int]): list of nodes to read 
            path_to_data (Path): path to where summaries are stored
            root_file (str): root file for summaries to be read
            filters (Dict): dictionary of filters to apply on the summaries' coordinates
        """
        self.nodes = nodes
        self.path_to_data = path_to_data
        self.root_file = root_file
        if filters is not None:
            self.filters = self.transform_filters_to_slices(filters)
        else:
            self.filters = None
        self.summaries = self.load()

    def transform_filters_to_slices(self, filters: Dict)->Dict:
        """Transform a dictionary of filters into slices that select from min to max

        Args:
            filters (Dict): dictionary of filters. Example:
                filters = {'r': (10,100)} , will select the summary statistics for 10 < r < 100

        Returns:
            Dict: dictionary of filters with slices
        """
        for filter, (min, max) in filters.items():
            filters[filter] = slice(min, max)
        return filters

    def load_summary(self, node: int,)->xr.DataArray:
        """Load the summary for a particular node

        Args:
            node (int): node to load

        Returns:
            xr.DataArray: data array with coordinates and summary value
        """
        summary = xr.open_dataarray(self.path_to_data / f'{self.root_file}_node{node}.nc')
        if self.filters:
            return summary.sel(**self.filters)
        return summary

    def load(self,)->np.array:
        """Load all summaries as a numpy array

        Returns:
            np.array: array of summaries
        """
        summaries = []
        for node in self.nodes:
            summaries.append(
                self.load_summary(node=node).to_numpy()
            )
        return np.array(summaries)