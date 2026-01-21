from copy import deepcopy
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .bin import Bin
from .constants import RT_ALL, BRKGA_Status, RotationType, RT_NotUpdown
from .item import Item
from .logger import log
from .painter import Painter


class PlacementProcedure:
    def __init__(self, bins: List[Bin], items: List[Item], solution: NDArray[np.float64], is_debug: bool = False) -> None:
        self.bins: List[Bin] = bins
        self.items: List[Item] = items
        self.infeasible_fitness: int = len(bins) + 1
        self.num_opened_bins: int = 1
        self.unpacked_items: List[Item] = []
        self.is_debug: bool = is_debug

        # Decode the gene in the solution.
        self.box_packing_sequence: NDArray[np.int_] = np.argsort(solution[: len(self.items)])
        self.vector_of_box_orientation: NDArray[np.float64] = solution[len(self.items) :]
        self.sorted_items: List[Item] = [self.items[index] for index in self.box_packing_sequence]

        log.debug(f"All sorted items:\n{[item.partno for item in self.items]}")
        log.debug(f"Box packing sequence(BPS):\n{self.box_packing_sequence.tolist()}")
        log.debug(f"Vector of box orientations(VBO):\n{self.vector_of_box_orientation.tolist()}")

        self.infeasible: bool = False  # Whether it is possible to place all items with the bin number.
        self._placeItem()

    def _placeItem(self) -> None:
        """
        Place items into bins.
        """
        for item_index, item in enumerate(self.sorted_items):
            log.info(f"Select box: {item.partno}.")
            # Bin and EMS selection.
            selected_bin_index = None
            selected_EMS = None
            is_placed = False
            for bin_index in range(self.num_opened_bins):
                # Select the remaining EMSs using distance to front-top-right center.
                for available_EMS in self.computeDistanceToFrontTopRightCenter(item, bin_index):
                    selected_bin_index = bin_index
                    selected_EMS = available_EMS
                    log.info(f"Select EMS {[ai.tolist() for ai in available_EMS]} from the [{bin_index}] bin.")
                    # Box orientation selection.
                    self.selectBoxOrientation(float(self.vector_of_box_orientation[item_index]), item, selected_EMS)
                    # Fix the item that flows in the air.
                    if self.bins[selected_bin_index].fixAirPlace(item, selected_EMS):
                        log.debug(f"Can place the [{item.partno}] in the EMS {available_EMS.tolist()}.")
                        is_placed = True
                        break
                if is_placed:
                    break

            # Open new empty bin.
            if not is_placed:
                if self.num_opened_bins + 1 > len(self.bins):
                    self.infeasible = True
                    self.unpacked_items.append(item)
                    log.warning(f"No more bin to open. Bin number is [{self.num_opened_bins}].")
                    continue

                selected_bin_index = self.num_opened_bins
                self.num_opened_bins += 1
                # Place the item into the origin of the new bin.
                selected_EMS = self.bins[selected_bin_index].EMSs[0]
                log.warning(f"No available bin, select the origin of the [{selected_bin_index}-th] new bin as the EMS.")

            # Elimination rule for different process.
            min_vol, min_dim = self.getMinVolAndDimOfRemainingItems(self.sorted_items[item_index + 1 :])
            log.debug(f"Minimal vol: [{min_vol}], minimal dim: [{min_dim}].")

            # pack the box to the bin & update state information
            assert selected_EMS is not None, f"{selected_EMS = }"
            assert selected_bin_index is not None, f"{selected_bin_index = }"
            self.bins[selected_bin_index].updateEMS(item, min_vol, min_dim)

            log.info(f"Add box to bin: {selected_bin_index}")
            log.info(f"EMSs:\n{self.bins[selected_bin_index].getEMSs()}")
            if self.is_debug:
                painter = Painter(self.bins[selected_bin_index])
                fig = painter.plotItemsAndBin(title=self.bins[selected_bin_index].partno, alpha=0.5, write_num=False, fontsize=10)
                plt = painter.plotRemainedEMS()
                plt.show()
        log.debug(f"Number of used bins: {self.num_opened_bins}")

    def computeDistanceToFrontTopRightCenter(self, item: Item, bin_index: int) -> List[NDArray[np.float_]]:
        """
        Compute the distance of the item to the front-top-right corner of the bin according to the DFRTC algorithm.
        However, the item may flow in the air. Therefore, return the available EMS as soon as possible once we find one.
        Because we will fix the position of the item after then.
        :param item: The item to place.
        :param bin_index: The index of the bin used to pack the item.
        :return: If the item can place into the bin, return the corresponding EMS of the used bin.
        """
        available_EMSs: List[NDArray[np.float64]] = []
        curr_EMSs = self.bins[bin_index].EMSs
        randon_index = np.random.choice(np.arange(0, curr_EMSs.shape[0]), size=curr_EMSs.shape[0], replace=False)
        for unplaced_items in deepcopy(curr_EMSs[randon_index]):  # Add more random otherwise would always explore the old EMS first.
            rotate = RT_ALL if item.updown == True else RT_NotUpdown
            for rt_type in rotate:
                item.rotation_type = rt_type
                if self.putItemIntoBin(item, unplaced_items):
                    available_EMSs.append(unplaced_items)
                    break  # If we want to find all available EMSs, the comment this line.
        sorted_available_EMSs = sorted(available_EMSs, key=lambda x: np.prod(x[3:] - x[:3]), reverse=True)
        return sorted_available_EMSs

    def selectBoxOrientation(self, vector_of_box_orientation: float, item: Item, selected_EMS: NDArray[np.float64]) -> None:
        """
        Select the box orientation from the available orientations.
        :param vector_of_box_orientation: The value between [0,1] which is used to choose the orientation.
        :param item: The box to place.
        :param selected_EMS: The empty maximum space where to place the item at its origin.
        """
        available_box_orientations: List[RotationType] = []
        rotate = RT_ALL if item.updown == True else RT_NotUpdown
        for rt_type in rotate:
            item.rotation_type = rt_type
            if self.putItemIntoBin(item, selected_EMS):
                available_box_orientations.append(rt_type)
        # Choose box orientation based on available box orientation vector.
        item.rotation_type = available_box_orientations[int(np.ceil(vector_of_box_orientation * len(available_box_orientations)) - 1)]
        log.info(f"Select VBO: {item.rotation_type} from {available_box_orientations = }, vector: {vector_of_box_orientation}.")

    @staticmethod
    def putItemIntoBin(box: Item, EMS: NDArray[np.float64]) -> bool:
        """
        Check whether the box can put into the EMS at its origin.
        :param box: Box to pack.
        :param EMS: The empty maximal space.
        """
        if np.any((np.array(box.getDimension()) - (EMS[3:] - EMS[:3])) > 0):
            return False
        return True

    @staticmethod
    def getMinVolAndDimOfRemainingItems(unplaced_items: List[Item]) -> Tuple[float, float]:
        """
        Compute the minimum volume and dimension of the unplaced items.
        :param unplaced_items: The unplaced items.
        """
        if len(unplaced_items) == 0:
            return 0.0, 0.0

        dimension = np.vstack([item.getDimension() for item in unplaced_items])
        min_dim = np.min(dimension)
        min_vol = np.min(np.prod(dimension, axis=1))
        return float(min_vol), float(min_dim)

    def evaluateSolution(self) -> float:
        """
        Evaluate the fitness of the solution.
        If it is impossible to place all items into the given bins, the fitness is equal to [bin number + 1 + unplaced volume ratio].
        If it is possible to place all items into the given bins, the fitness is equal to [open bin umber + 1 - the least placed volume ratio of the bin].
        # FIXME: Distinguish the cases which can put all item into the bins to find a  better solution.
        """
        if self.infeasible:
            unpacked_Volume_ratio = float(np.sum([unpacked_item.getVolume() for unpacked_item in self.unpacked_items])) / float(np.sum([bin.getVolume() for bin in self.bins]))
            return self.infeasible_fitness + unpacked_Volume_ratio

        leastLoad = 1.0
        for bin_index in range(self.num_opened_bins):
            load = self.bins[bin_index].calculateUtilizationRate()
            if load < leastLoad:
                leastLoad = load
        return self.num_opened_bins + 1 - leastLoad


class BRKGA:
    def __init__(
        self,
        bins: List[Bin],
        items: List[Item],
        num_generations: int = 200,
        num_individuals: int = 120,
        num_elites: int = 12,
        num_mutants: int = 18,
        eliteCProb: float = 0.7,
    ) -> None:
        # Input
        self.bins: List[Bin] = bins
        self.items: List[Item] = items
        self.num_item: int = len(items)

        self.num_generations: int = num_generations
        self.num_individuals: int = int(num_individuals)
        # The first n genes encode the order of the n items to be packed, which is called Box Packing Sequence (box_packing_sequence).
        # Decode it by sorting it in ascending order of the corresponding gene value.
        # The last n genes encode the orientation of the n item, which is called Vector of Box Orientation (VBO).
        # Decode it by the multiplt it with the length of the number of orientation.
        self.num_gene: int = 2 * self.num_item
        self.num_elites: int = int(num_elites)
        self.num_mutants: int = int(num_mutants)
        assert 0 <= eliteCProb <= 1, f"The value should be between [0, 1]."
        # The prespecified probability which decides the i-th gene is inherited from the elite or the non-elite parents.
        self.eliteCProb: float = eliteCProb

        # Result
        self.best_solution: NDArray[np.float64] = np.array([])
        self.best_fitness: float = -1.0
        self.best_bins: List[Bin] = []
        self.best_unpacked_item: List[Item] = []
        self.fitness_mean_history: List[float] = []
        self.fitness_min_history: List[float] = []

    def calFitness(self, population: NDArray[np.float64]) -> Tuple[List[float], List[List[Bin]], List[List[Item]]]:
        """
        Calculate the fitness of all the individuals by decoding the biased random key element.
        :param population: Each row represents an individual.
        :return: The fitness of the population.
        """
        fitness_list: List[float] = []
        bins_list: List[List[Bin]] = []
        unpacked_item_list: List[List[Item]] = []
        for solution in population:
            decoder = PlacementProcedure(deepcopy(self.bins), deepcopy(self.items), solution)
            fitness_list.append(decoder.evaluateSolution())
            bins_list.append(deepcopy(decoder.bins[: decoder.num_opened_bins]))
            unpacked_item_list.append(deepcopy(decoder.unpacked_items))
        return fitness_list, bins_list, unpacked_item_list

    def separatePopulation(
        self,
        population: NDArray[np.float64],
        fitness_list: List[float],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[float]]:
        """
        Separate the population into elite and non-elite subpopulation in order. Calculate the fitness of the elite subpopulation.
        :param population: The population.
        :param fitness_list: The fitness of the population.
        """
        sorted_index = np.argsort(fitness_list)
        return population[sorted_index[: self.num_elites]], population[sorted_index[self.num_elites :]], np.array(fitness_list)[sorted_index[: self.num_elites]].tolist()

    def crossover(self, elite: NDArray[np.float64], non_elite: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Choose the gene from either elite or non_elite parents for each gene of the offspring according to the elite probability.
        :param elite: The elite individual.
        :param non_elite: The non-elite individual.
        :return: The offspring generated by crossover.
        """
        return np.array([elite[gene_index] if np.random.uniform(low=0.0, high=1.0) < self.eliteCProb else non_elite[gene_index] for gene_index in range(self.num_gene)])

    def mating(self, elites: NDArray[np.float64], non_elites: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Generate all the offsprings by crossover from the elite and non-elite parents.
        :param elites: The elite subpopulation.
        :param non_elites: The non-elite subpopulation.
        :return: All the offsprings by crossover.
        """
        num_offspring = self.num_individuals - self.num_elites - self.num_mutants
        return np.array([self.crossover(elites[np.random.choice(elites.shape[0])], non_elites[np.random.choice(non_elites.shape[0])]) for _ in range(num_offspring)])

    def mutants(self) -> NDArray[np.float64]:
        """
        Generate the mutations to increase the variegation in the offsprings.
        :return: All the offsprings by mutation.
        """
        return np.random.uniform(low=0.0, high=1.0, size=(self.num_mutants, self.num_gene))

    def _initializeBRKGA(self) -> Tuple[NDArray[np.float64], List[float], List[List[Bin]], List[List[Item]]]:
        """
        Initialize the population and fitness.
        """
        population = np.random.uniform(low=0.0, high=1.0, size=(self.num_individuals, self.num_gene))  # Each row represents an individual.
        fitness_list, bin_list, unpacked_item_list = self.calFitness(population)
        log.info(f"Initial Population: shape = [{population.shape}], Best Fitness = [{max(fitness_list)}].")
        return population, fitness_list, bin_list, unpacked_item_list

    def fit(self, patient: int = 4) -> BRKGA_Status:
        """
        Generate the 3D bin packing result by biased random key generic algorithm (BRKGA).
        :param patient: When the current generation index minus best generation index is larget than the patient index, it means the iteration has reach local optimization.
        :return: Biased random key generic algorithm result.
        """
        population, fitness_list, bin_list, unpacked_item_list = self._initializeBRKGA()

        # Initialize the best configuration.
        best_fitness = np.min(fitness_list)
        best_solution = population[np.argmin(fitness_list)]
        best_bin = bin_list[np.argmin(fitness_list)]
        best_unpacked_item = unpacked_item_list[np.argmin(fitness_list)]
        self.fitness_min_history.append(np.min(fitness_list))
        self.fitness_mean_history.append(float(np.mean(fitness_list)))

        # Repeat generations.
        best_iter = 0
        for generation_index in range(self.num_generations):
            # Early stopping because it is hard to find a better solution than the current best solution (local optimization).
            if generation_index - best_iter > patient:
                log.warning(f"Early stop at iter: {generation_index}, because it is hard to find a better solution than the current best solution.")
                break

            # Select elite subpopulation.
            elites, non_elites, elite_fitness_list = self.separatePopulation(population, fitness_list)

            # Biased mating by crossover.
            offsprings = self.mating(elites, non_elites)

            # Generate mutation.
            mutants = self.mutants()

            # Concatenate the new population and calculate the fitness.
            population = np.concatenate((elites, mutants, offsprings), axis=0)
            fitness_list, bin_list, unpacked_item_list = self.calFitness(population)

            # Update best fitness.
            for fitness, used_bin, unpacked_item in zip(fitness_list, bin_list, unpacked_item_list):
                if fitness < best_fitness:
                    best_iter = generation_index
                    best_fitness = fitness
                    best_solution = population[np.argmin(fitness_list)]
                    best_bin = used_bin
                    best_unpacked_item = unpacked_item

            self.fitness_min_history.append(np.min(fitness_list))
            self.fitness_mean_history.append(float(np.mean(fitness_list)))

            log.info(f"Generation = [{generation_index}], best fitness = [{best_fitness}]")

        self.best_fitness = best_fitness
        self.best_solution = best_solution
        self.best_bins = best_bin
        self.best_unpacked_item = best_unpacked_item

        return BRKGA_Status(is_succeed=True, msg="Finished iteration.")
