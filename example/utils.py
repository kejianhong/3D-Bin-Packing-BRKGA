from typing import List, Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from brkga import BRKGA, Painter, log


def PlotGenerationProcess(fitness_mean: List[float], fitness_min: List[float], filename: Optional[str]=None) -> None:
    plt.figure()
    plt.plot(np.linspace(1, len(fitness_min) + 1, len(fitness_min)), fitness_min, color="red", label="minimal fitness")
    plt.plot(np.linspace(1, len(fitness_min) + 1, len(fitness_min)), fitness_mean, color="blue", label="mean fitness")
    ax = plt.gca()
    # ax.set_aspect("equal")
    plt.grid(True)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Value")
    ax.legend()
    if filename is not None:
        plt.savefig(f"{filename}.png")
    else:
        plt.show()


def PrintPackingResult(brkga: BRKGA, filename: Optional[tuple[str, str]] = None) -> None:
    volume_f = 0.0
    unfit_items_string = "UNFITTED ITEMS:\n"
    for item in brkga.best_unpacked_item:
        unfit_items_string = f"{unfit_items_string}{item.string()}\n"
        volume_f += item.getVolume()
    log.debug(f"{unfit_items_string}")
    log.info(f"unpack item volume: {volume_f}")

    result_name, ems_name = (None, None) if filename is None else filename
    for index, bin in enumerate(brkga.best_bins):
        volume = bin.getVolume()
        fit_items_string = "FITTED ITEMS:\n"
        volume_t = 0.0
        for item in bin.items:
            fit_items_string = f"{fit_items_string}{item.string()}\n"
            volume_t += item.getVolume()
        log.debug(f"{fit_items_string}")
        log.info(f"space utilization: {round(volume_t / volume * 100, 2)}")
        log.info(f"residual volume: {volume - volume_t}")

        painter = Painter(bin)
        fig = painter.plotItemsAndBin(title=bin.partno, alpha=0.5, write_num=False, fontsize=10, filename=f"{result_name}_{index}.png" if result_name else None)
        fig = painter.plotRemainedEMS(filename=f"{ems_name}_{index}.png" if ems_name else None)
    if filename is None:
        fig.show()
