import logging
import time
from typing import List

import numpy as np

from brkga import BRKGA, Bin, Item, log

from .utils import PlotGenerationProcess, PrintPackingResult


def example1() -> None:
    """
    Example 1: one bin with different items.
    """
    bins: List[Bin] = [
        Bin(partno="example1", WHD=(5.6875, 10.75, 15.0), support_surface_ratio=0.75),
    ]

    items: List[Item] = [
        Item("50g [powder 1]", "test", "cube", (2, 2, 4), 1, 1, 100, True, "red"),
        Item("50g [powder 2]", "test", "cube", (2, 2, 4), 2, 1, 100, True, "blue"),
        Item("50g [powder 3]", "test", "cube", (2, 2, 4), 3, 1, 100, True, "gray"),
        Item("50g [powder 4]", "test", "cube", (2, 2, 4), 3, 1, 100, True, "orange"),
        Item("50g [powder 5]", "test", "cube", (2, 2, 4), 3, 1, 100, True, "lawngreen"),
        Item("50g [powder 6]", "test", "cube", (2, 2, 4), 3, 1, 100, True, "purple"),
        Item("50g [powder 7]", "test", "cube", (1, 1, 5), 3, 1, 100, True, "yellow"),
        Item("250g [powder 8]", "test", "cube", (4, 4, 2), 4, 1, 100, True, "pink"),
        Item("250g [powder 9]", "test", "cube", (4, 4, 2), 5, 1, 100, True, "brown"),
        Item("250g [powder 10]", "test", "cube", (4, 4, 2), 6, 1, 100, True, "cyan"),
        Item("250g [powder 11]", "test", "cube", (4, 4, 2), 7, 1, 100, True, "olive"),
        Item("250g [powder 12]", "test", "cube", (4, 4, 2), 8, 1, 100, True, "darkgreen"),
        Item("250g [powder 13]", "test", "cube", (4, 4, 2), 9, 1, 100, True, "orange"),
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
    brkga.fit(patient=5)
    stop_time = time.time()
    PrintPackingResult(brkga)
    # PlacementProcedure(brkga.bins, brkga.items, brkga.best_solution, True)
    log.warning(f"Used time: {stop_time - start_time}")
    log.warning(f"Best fitness: {brkga.best_fitness}")
    PlotGenerationProcess(brkga.fitness_mean_history, brkga.fitness_min_history)


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    np.random.seed(123)
    example1()
