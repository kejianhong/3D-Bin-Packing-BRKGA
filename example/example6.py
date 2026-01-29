import logging
import time
from typing import List, Tuple

import numpy as np

from brkga import BRKGA, Bin, Item, PlacementProcedure, log

from .utils import PlotGenerationProcess, PrintPackingResult


def example6() -> None:
    """
    Example 6: one bin with different items.
    """
    bins: List[Bin] = [
        Bin(partno="example6", WHD=(5, 4, 7), support_surface_ratio=0.75),
    ]

    items: List[Item] = [
        Item(partno="Box-1", name="test", typeof="cube", WHD=(5, 4, 1), weight=1, level=1, loadbear=100, updown=True, color="yellow"),
        Item(partno="Box-2", name="test", typeof="cube", WHD=(1, 1, 4), weight=1, level=2, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-3", name="test", typeof="cube", WHD=(3, 4, 2), weight=1, level=3, loadbear=100, updown=True, color="pink"),
        Item(partno="Box-4", name="test", typeof="cube", WHD=(1, 1, 4), weight=1, level=4, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-5", name="test", typeof="cube", WHD=(1, 2, 1), weight=1, level=5, loadbear=100, updown=True, color="pink"),
        Item(partno="Box-6", name="test", typeof="cube", WHD=(1, 2, 1), weight=1, level=6, loadbear=100, updown=True, color="pink"),
        Item(partno="Box-7", name="test", typeof="cube", WHD=(1, 1, 4), weight=1, level=7, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-8", name="test", typeof="cube", WHD=(1, 1, 4), weight=1, level=8, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-9", name="test", typeof="cube", WHD=(5, 4, 2), weight=1, level=9, loadbear=100, updown=True, color="brown"),
    ]

    start_time = time.time()
    brkga = BRKGA(
        bins=bins,
        items=items,
        num_generations=70,
        num_individuals=70,
        num_elites=30,
        num_mutants=20,
        eliteCProb=0.7,
    )
    brkga.fit(patient=15)
    stop_time = time.time()

    PrintPackingResult(brkga, filename=("example6_result", "example6_ems"))
    log.warning(f"Used time: {stop_time - start_time}")
    log.warning(f"Best fitness: {brkga.best_fitness}")
    PlotGenerationProcess(brkga.fitness_mean_history, brkga.fitness_min_history, filename="example6_generation_process")

if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    np.random.seed(123)
    example6()
