import logging
import time
from typing import List, Tuple

import numpy as np

from brkga import BRKGA, Bin, Item, PlacementProcedure, log

from .utils import PlotGenerationProcess, PrintPackingResult


def example7() -> None:
    """
    Example 7: one bin with different items.
    """
    bins: List[Bin] = [
        Bin(partno="example7-Bin1", WHD=(5, 5, 5), support_surface_ratio=0.75),
        Bin(partno="example7-Bin2", WHD=(5, 5, 5), support_surface_ratio=0.75),
    ]

    items: List[Item] = [
        Item(partno="Box-1", name="test1", typeof="cube", WHD=(5, 4, 1), weight=1, level=1, loadbear=100, updown=True, color="yellow"),
        Item(partno="Box-2", name="test2", typeof="cube", WHD=(1, 2, 4), weight=1, level=1, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-3", name="test3", typeof="cube", WHD=(1, 2, 3), weight=1, level=1, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-4", name="test4", typeof="cube", WHD=(1, 2, 2), weight=1, level=1, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-5", name="test5", typeof="cube", WHD=(1, 2, 3), weight=1, level=1, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-6", name="test6", typeof="cube", WHD=(1, 2, 4), weight=1, level=1, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-7", name="test7", typeof="cube", WHD=(1, 2, 2), weight=1, level=1, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-8", name="test8", typeof="cube", WHD=(1, 2, 3), weight=1, level=1, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-9", name="test9", typeof="cube", WHD=(1, 2, 4), weight=1, level=1, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-10", name="test10", typeof="cube", WHD=(1, 2, 3), weight=1, level=1, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-11", name="test11", typeof="cube", WHD=(1, 2, 2), weight=1, level=1, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-12", name="test12", typeof="cube", WHD=(5, 4, 1), weight=1, level=1, loadbear=100, updown=True, color="pink"),
        Item(partno="Box-13", name="test13", typeof="cube", WHD=(1, 1, 4), weight=1, level=1, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-14", name="test14", typeof="cube", WHD=(1, 2, 1), weight=1, level=1, loadbear=100, updown=True, color="pink"),
        Item(partno="Box-15", name="test15", typeof="cube", WHD=(1, 2, 1), weight=1, level=1, loadbear=100, updown=True, color="pink"),
        Item(partno="Box-16", name="test16", typeof="cube", WHD=(1, 1, 4), weight=1, level=1, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-17", name="test17", typeof="cube", WHD=(1, 1, 4), weight=1, level=1, loadbear=100, updown=True, color="olive"),
        Item(partno="Box-18", name="test18", typeof="cube", WHD=(5, 4, 2), weight=1, level=1, loadbear=100, updown=True, color="brown"),
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

    PrintPackingResult(brkga)
    log.warning(f"Used time: {stop_time - start_time}")
    log.warning(f"Best fitness: {brkga.best_fitness}")
    PlotGenerationProcess(brkga.fitness_mean_history, brkga.fitness_min_history)


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    np.random.seed(123)
    example7()
