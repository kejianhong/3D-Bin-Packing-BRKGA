import logging
import time
from typing import List

import numpy as np

from brkga import BRKGA, Bin, Item, log

from .utils import PlotGenerationProcess, PrintPackingResult


def example3() -> None:
    """
    Example 3: one bin with different items.
    """
    bins: List[Bin] = [
        Bin(partno="example3", WHD=(6, 1, 5), support_surface_ratio=0.75),
    ]

    items: List[Item] = [
        Item(partno="Box-1", name="test", typeof="cube", WHD=(2, 1, 3), weight=1, level=1, loadbear=100, updown=True, color="yellow"),
        Item(partno="Box-2", name="test", typeof="cube", WHD=(3, 1, 2), weight=1, level=1, loadbear=100, updown=True, color="pink"),
        Item(partno="Box-3", name="test", typeof="cube", WHD=(2, 1, 3), weight=1, level=1, loadbear=100, updown=True, color="brown"),
        Item(partno="Box-4", name="test", typeof="cube", WHD=(2, 1, 3), weight=1, level=1, loadbear=100, updown=True, color="cyan"),
        Item(partno="Box-5", name="test", typeof="cube", WHD=(2, 1, 3), weight=1, level=1, loadbear=100, updown=True, color="olive"),
    ]

    start_time = time.time()
    brkga = BRKGA(
        bins=bins,
        items=items,
        num_generations=70,
        num_individuals=len(items),
        num_elites=int(len(items) / 2),
        num_mutants=int(len(items) / 4),
        eliteCProb=0.7,
    )
    brkga.fit(patient=15)
    stop_time = time.time()

    PrintPackingResult(brkga, filename=("example3_result", "example3_ems"))
    # PlacementProcedure(brkga.bins, brkga.items, brkga.best_solution, True)
    log.warning(f"Used time: {stop_time - start_time}")
    log.warning(f"Best fitness: {brkga.best_fitness}")
    PlotGenerationProcess(brkga.fitness_mean_history, brkga.fitness_min_history, filename="example3_generation_process")

if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    np.random.seed(123)
    example3()
