from abc import ABC, abstractmethod
import numpy as np
from summarizer.data import Catalogue


class BaseSummary(ABC):
    @abstractmethod
    def __call__(self, catalogue: Catalogue) -> np.array:
        """Given a catalogue, produce a summary

        Args:
            catalogue (Catalogue): catalogue to summarize

        Returns:
            np.array: summary 
        """
        return

    @abstractmethod
    def store_summary(self, summary: np.array):
        """Store the results from summarization to file

        Args:
            summary (np.array): summary to store
        """
        pass
