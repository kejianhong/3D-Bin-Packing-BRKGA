import logging
import time
from typing import List

import numpy as np

from brkga import BRKGA, Bin, Item, log

from .utils import PlotGenerationProcess, PrintPackingResult


def example4() -> None:
    """
    Example 4: one bin with different items.
    """
    bins: List[Bin] = [
        Bin(partno="example4", WHD=(589.8, 243.8, 259.1), support_surface_ratio=0.75),
    ]

    items: List[Item] = []

    # dyson DC34 (20.5 * 11.5 * 32.2 ,1.33kg)
    # 64 pcs per case ,  82 * 46 * 170 (85.12)
    for i in range(15):
        items.append(
            Item(
                partno="Dyson DC34 Animal{}".format(str(i + 1)),
                name="Dyson",
                typeof="cube",
                WHD=(170, 82, 46),
                weight=85.12,
                level=1,
                loadbear=100,
                updown=True,
                color="#FF0000",
            )
        )

    # washing machine (85 * 60 *60 ,10 kG)
    # 1 pcs per case, 85 * 60 *60 (10)
    for i in range(18):
        items.append(
            Item(
                partno="wash{}".format(str(i + 1)),
                name="wash",
                typeof="cube",
                WHD=(85, 60, 60),
                weight=10,
                level=1,
                loadbear=100,
                updown=True,
                color="#FFFF37",
            )
        )

    # 42U standard cabinet (60 * 80 * 200 , 80 kg)
    # 1 per box, 60 * 80 * 200 (80)
    for i in range(15):
        items.append(
            Item(
                partno="Cabinet{}".format(str(i + 1)),
                name="cabint",
                typeof="cube",
                WHD=(60, 80, 200),
                weight=80,
                level=1,
                loadbear=100,
                updown=True,
                color="#842B00",
            )
        )

    # Server (70 * 100 * 30 , 20 kg)
    # 1 per box , 70 * 100 * 30 (20)
    for i in range(42):
        items.append(
            Item(
                partno="Server{}".format(str(i + 1)),
                name="server",
                typeof="cube",
                WHD=(70, 100, 30),
                weight=20,
                level=1,
                loadbear=100,
                updown=True,
                color="#0000E3",
            )
        )
    start_time = time.time()
    brkga = BRKGA(
        bins=bins,
        items=items,
        num_generations=70,
        num_individuals=int(len(items) / 2),
        num_elites=int(len(items) / 4),
        num_mutants=int(len(items) / 6),
        eliteCProb=0.7,
    )
    brkga.fit(patient=5)
    stop_time = time.time()

    PrintPackingResult(brkga)
    log.warning(f"Used time: {stop_time - start_time}")
    log.warning(f"Best fitness: {brkga.best_fitness}")
    PlotGenerationProcess(brkga.fitness_mean_history, brkga.fitness_min_history)


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    np.random.seed(123)
    example4()
