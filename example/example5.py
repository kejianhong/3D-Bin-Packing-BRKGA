import logging
import time
from typing import List

import numpy as np

from brkga import BRKGA, Bin, Item, log

from .utils import PlotGenerationProcess, PrintPackingResult


def example5() -> None:
    """
    Example 5: one bin with different items.
    """
    bins: List[Bin] = [
        Bin(partno="example5", WHD=(5, 4, 3), support_surface_ratio=0.75),
    ]

    items: List[Item] = [
        Item(partno="Box-2", name="test", typeof="cube", WHD=(2, 5, 2), weight=1, level=1, loadbear=100, updown=True, color="pink"),
        Item(partno="Box-3", name="test", typeof="cube", WHD=(2, 3, 2), weight=1, level=2, loadbear=100, updown=True, color="purple"),
        Item(partno="Box-4", name="test", typeof="cube", WHD=(5, 4, 1), weight=1, level=3, loadbear=100, updown=True, color="brown"),
    ]

    start_time = time.time()
    brkga = BRKGA(
        bins=bins,
        items=items,
        num_generations=10,
        num_individuals=10,
        num_elites=5,
        num_mutants=2,
        eliteCProb=0.7,
    )
    brkga.fit(patient=7)
    stop_time = time.time()

    PrintPackingResult(brkga, filename=("example5_result", "example5_ems"))
    # PlacementProcedure(brkga.bins, brkga.items, brkga.best_solution, True)
    log.warning(f"Used time: {stop_time - start_time}")
    log.warning(f"Best fitness: {brkga.best_fitness}")
    PlotGenerationProcess(brkga.fitness_mean_history, brkga.fitness_min_history, filename="example5_generation_process")

if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    np.random.seed(123)
    example5()
