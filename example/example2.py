import logging
import time
from typing import List

import numpy as np

from brkga import BRKGA, Bin, Item, log

from .utils import PlotGenerationProcess, PrintPackingResult


def example2() -> None:
    """
    Example 2: one bin with different items.
    """
    bins: List[Bin] = [
        Bin(partno="example2", WHD=(30, 10, 15), support_surface_ratio=0.75),
    ]

    items: List[Item] = [
        Item("test1", "test", "cube", (9, 8, 7), 1, 1, 100, True, "red"),
        Item("test2", "test", "cube", (4, 25, 1), 1, 1, 100, True, "blue"),
        Item("test3", "test", "cube", (2, 13, 5), 1, 1, 100, True, "gray"),
        Item("test4", "test", "cube", (7, 5, 4), 1, 1, 100, True, "orange"),
        Item("test5", "test", "cube", (10, 5, 2), 1, 1, 100, True, "lawngreen"),
        Item("test6", "test", "cube", (6, 5, 2), 1, 1, 100, True, "purple"),
        Item("test7", "test", "cube", (5, 2, 9), 1, 1, 100, True, "yellow"),
        Item("test8", "test", "cube", (10, 8, 5), 1, 1, 100, True, "pink"),
        Item("test9", "test", "cube", (1, 3, 5), 1, 1, 100, True, "brown"),
        Item("test10", "test", "cube", (8, 4, 7), 1, 1, 100, True, "cyan"),
        Item("test11", "test", "cube", (2, 5, 3), 1, 1, 100, True, "olive"),
        Item("test12", "test", "cube", (1, 9, 2), 1, 1, 100, True, "darkgreen"),
        Item("test13", "test", "cube", (7, 5, 4), 1, 1, 100, True, "orange"),
        Item("test14", "test", "cube", (10, 2, 1), 1, 1, 100, True, "lawngreen"),
        Item("test15", "test", "cube", (3, 2, 4), 1, 1, 100, True, "purple"),
        Item("test16", "test", "cube", (5, 7, 8), 1, 1, 100, True, "yellow"),
        Item("test17", "test", "cube", (4, 8, 3), 1, 1, 100, True, "white"),
        Item("test18", "test", "cube", (2, 11, 5), 1, 1, 100, True, "brown"),
        Item("test19", "test", "cube", (8, 3, 5), 1, 1, 100, True, "cyan"),
        Item("test20", "test", "cube", (7, 4, 5), 1, 1, 100, True, "olive"),
        Item("test21", "test", "cube", (2, 4, 11), 1, 1, 100, True, "darkgreen"),
        Item("test22", "test", "cube", (1, 3, 4), 1, 1, 100, True, "orange"),
        Item("test23", "test", "cube", (10, 5, 2), 1, 1, 100, True, "lawngreen"),
        Item("test24", "test", "cube", (7, 4, 5), 1, 1, 100, True, "purple"),
        Item("test25", "test", "cube", (2, 10, 3), 1, 1, 100, True, "yellow"),
        Item("test26", "test", "cube", (3, 8, 1), 1, 1, 100, True, "pink"),
        Item("test27", "test", "cube", (7, 2, 5), 1, 1, 100, True, "brown"),
        Item("test28", "test", "cube", (8, 9, 5), 1, 1, 100, True, "cyan"),
        Item("test29", "test", "cube", (4, 5, 10), 1, 1, 100, True, "olive"),
        Item("test30", "test", "cube", (10, 10, 2), 1, 1, 100, True, "darkgreen"),
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
    PrintPackingResult(brkga, filename=("example2_result", "example2_ems"))
    log.warning(f"Used time: {stop_time - start_time}")
    log.warning(f"Best fitness: {brkga.best_fitness}")
    PlotGenerationProcess(brkga.fitness_mean_history, brkga.fitness_min_history, filename="example2_generation_process")


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    np.random.seed(123)
    example2()
