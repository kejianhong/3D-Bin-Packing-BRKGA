import dataclasses
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .constants import DELTA
from .item import Item
from .logger import log


@dataclass
class Rectangle:
    left_back_x: float
    left_back_y: float
    right_front_x: float
    right_front_y: float
    center_x: float = dataclasses.field(init=False)
    center_y: float = dataclasses.field(init=False)
    length: float = dataclasses.field(init=False)
    width: float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.center_x = (self.left_back_x + self.right_front_x) / 2.0
        self.center_y = (self.left_back_y + self.right_front_y) / 2.0
        self.length = max(self.left_back_x, self.right_front_x) - min(self.left_back_x, self.right_front_x)
        self.width = max(self.left_back_y, self.right_front_y) - min(self.left_back_y, self.right_front_y)


def checkIntersect(rect1: Rectangle, rect2: Rectangle) -> bool:
    """
    Check whether two rectangles are intersected with each other or not.
    """
    center_distance_x = max(rect1.center_x, rect2.center_x) - min(rect1.center_x, rect2.center_x)
    center_distance_y = max(rect1.center_y, rect2.center_y) - min(rect1.center_y, rect2.center_y)
    return center_distance_x < (rect1.length + rect2.length) / 2.0 and center_distance_y < (rect1.width + rect2.width) / 2.0


def combineLineSegment(line_segment: List[List[float]]) -> List[List[float]]:
    """
    Merge the line segments if they are intersected. O(nlog(n)) for the sort algorithm.
    """
    line_segment.sort(key=lambda line: line[0])
    merge_line_segment = [line_segment[0]]
    for index in range(1, len(line_segment)):
        last_line = merge_line_segment[-1]
        curr_line = line_segment[index]
        if curr_line[0] <= last_line[1]:
            merge_line = [last_line[0], max(last_line[1], curr_line[1])]
            merge_line_segment[-1] = merge_line
        else:
            merge_line_segment.append(curr_line)
    return merge_line_segment


class Bin:
    def __init__(self, partno: str, WHD: Tuple[float, float, float], support_surface_ratio: float) -> None:
        self.partno = partno
        self.items: List[Item] = []
        self.width: float = WHD[0]
        self.height: float = WHD[1]
        self.depth: float = WHD[2]
        self.dimensions: Tuple[float, float, float] = WHD
        self.EMSs: List[List[NDArray[np.float_]]] = [[np.array((0, 0, 0)), np.array(WHD)]]  # The ems (left-back-down, right-front-up) of the empty maximal space.
        self.fit_items: NDArray[np.float_] = np.array([[0, WHD[0], 0, WHD[1], 0, 0]])  # The ems (left-back-down, right-front-up) of the items placed in the bin.
        assert 0 < support_surface_ratio < 1, f"Should be in [0, 1]."
        self.support_surface_ratio = support_surface_ratio

        log.info(f"Init EMSs: {self.EMSs}")

    def __getitem__(self, index: int) -> List[NDArray[np.float_]]:
        return self.EMSs[index]

    def __len__(self) -> int:
        return len(self.EMSs)

    def getVolume(self) -> float:
        return self.width * self.height * self.depth

    def update(
        self,
        box: Item,
        min_vol: int = 1,
        min_dim: int = 1,
    ) -> None:
        """
        Update unused EMS of the bin according to the box.
        :param box: The dimension of the box which can pack into the bin.
        :param min_vol: Minimal volume of all the unpacked boxes.
        :param min_dim: Minimal dimension of all the unpacked boxes.
        """
        # 1. Place box in an EMS.
        x, y, z = box.position  # Should not use the `selected_EMS` bacause we modify the position of the item.
        w, h, d = box.getDimension()
        self.items.append(box)
        item_ems = [np.array([x, y, z]), np.array([x + w, y + h, z + d])]
        log.info(f"EMS of the packed items: {list(map(tuple, item_ems))}")

        # 2. Generate new EMSs resulting from the intersection of the box.
        all_EMS_index = list(range(len(self.EMSs)))
        overlapped_EMS_index: List[int] = []
        new_EMS_list: List[List[NDArray[np.float_]]] = []
        original_EMSs = deepcopy(self.EMSs)
        for unused_EMS_index, unused_EMS in enumerate(original_EMSs):  # Should use `deepcopy` because we would modify the `self.EMS` in the for loop.
            if self.checkInscribed(unused_EMS, item_ems):
                overlapped_EMS_index.append(unused_EMS_index)
                self.eliminateEMS(list(set(all_EMS_index) - set(overlapped_EMS_index)), original_EMSs)
                continue
            if self.checkOverlapped(item_ems=item_ems, unused_EMS=unused_EMS):
                # Eliminate overlapped EMS.
                overlapped_EMS_index.append(unused_EMS_index)
                self.eliminateEMS(list(set(all_EMS_index) - set(overlapped_EMS_index)), original_EMSs)
                log.info(f"Remove overlapped EMS: {list(map(tuple, unused_EMS))}\nEMSs left: {list(map(lambda x: list(map(tuple, x)), self.EMSs))}")

                # Add 6 new EMSs in 3 dimensions.
                x1, y1, z1 = unused_EMS[0]
                x2, y2, z2 = unused_EMS[1]
                x3, y3, z3 = item_ems[0]
                x4, y4, z4 = item_ems[1]
                new_EMSs = [
                    [np.array((x4, y1, z1)), np.array((x2, y2, z2))],  # front EMS
                    [np.array((x1, y4, z1)), np.array((x2, y2, z2))],  # right EMS
                    [np.array((x1, y1, z4)), np.array((x2, y2, z2))],  # up EMS
                    [np.array((x1, y1, z1)), np.array((x3, y2, z2))],  # back EMS
                    [np.array((x1, y1, z1)), np.array((x2, y3, z2))],  # left EMS
                    [np.array((x1, y1, z1)), np.array((x2, y2, z3))],  # down EMS
                ]
                for new_EMS in new_EMSs:
                    log.debug(f"New EMS {new_EMS}.")
                    new_EMS_size = new_EMS[1] - new_EMS[0]
                    if np.any(new_EMS_size <= 0):
                        log.debug(f"Impossible new EMS {new_EMS_size.tolist()} because there must be at least one dimension is negative.")
                        continue

                    # 3. Eliminate new EMSs which are totally inscribed by other EMSs.
                    is_inscribed = False
                    for other_EMS in self.EMSs:
                        if self.checkInscribed(new_EMS, other_EMS):
                            is_inscribed = True
                            log.info(f"New EMS [{list(map(tuple, new_EMS))}] is totally inscribed by [{list(map(tuple, other_EMS))}]")
                            break
                    if is_inscribed:
                        continue

                    # 4. Do not add new EMS smaller than the volume of remaining boxes
                    if np.min(new_EMS_size) < min_dim:
                        log.info(f"New EMS size = {new_EMS_size}, whose dimension too small than {min_dim}.")
                        continue

                    # 5. Do not add new EMS having smaller dimension of the smallest dimension of remaining boxes
                    if np.prod(new_EMS_size) < min_vol:
                        log.info(f"New EMS size = {new_EMS_size}, whose volume too small than {min_vol}.")
                        continue

                    new_EMS_list.append(new_EMS)
                    log.info(f"Successfully add new EMS: {list(map(tuple, new_EMS))}")

        self.EMSs = self.EMSs + new_EMS_list
        log.warning(f"Remaining {len(self.EMSs)} EMSs:\n{self.getEMSs()}")

    @staticmethod
    def checkOverlapped(item_ems: List[NDArray[np.float_]], unused_EMS: List[NDArray[np.float_]]) -> bool:
        """
        Check whether two EMSs are overlapped with each other.
        :param item_ems: The ems of the packed item.
        :param unused_EMS: The unused EMS of the bin.
        """
        if np.all(item_ems[1] >= unused_EMS[0]) and np.all(item_ems[0] <= unused_EMS[1]):
            return True
        return False

    @staticmethod
    def checkInscribed(EMS1: List[NDArray[np.float_]], EMS2: List[NDArray[np.float_]]) -> bool:
        """
        Check whether EMS1 is inscribed by EMS2.
        :param EMS1: The first EMS.
        :param EMS2: The second EMS.
        """
        if np.all(EMS2[0] <= EMS1[0]) and np.all(EMS1[1] <= EMS2[1]):
            return True
        return False

    def eliminateEMS(self, indexes: List[int], original_EMS: List[List[NDArray[np.float_]]]) -> None:
        """
        Eliminate EMS which is overlapped with the box packed into the bin.
        :param indexes:
        :param original_EMS:
        """
        self.EMSs = [original_EMS[i] for i in indexes]

    def getEMSs(self) -> List[Tuple[int, ...]]:
        return list(map(lambda x: tuple(x[1] - x[0]), self.EMSs))

    def calculateUtilizationRate(self) -> float:
        """
        Calculate the utilization rate of the bin.
        """
        return float(np.sum([np.prod(item[1::2] - item[::2]) for item in self.fit_items]) / np.prod(self.dimensions))

    def checkDepth(self, unfix_point: List[float]) -> Tuple[float, bool]:
        """
        Fix item position z.
        :param unfix_point: The back-left-down and front-right-up point coordinate of the item. [x_min, x_max, y_min, y_max, z_min, z_max].
        """
        z_: List[List[float]] = [[0, 0], [self.depth, self.depth]]
        for fix_item_corner in self.fit_items:
            rect1 = Rectangle(left_back_x=unfix_point[0], left_back_y=unfix_point[2], right_front_x=unfix_point[1], right_front_y=unfix_point[3])
            rect2 = Rectangle(left_back_x=fix_item_corner[0], left_back_y=fix_item_corner[2], right_front_x=fix_item_corner[1], right_front_y=fix_item_corner[3])
            if checkIntersect(rect1, rect2):
                z_.append([fix_item_corner[4], fix_item_corner[5]])
        top_depth = unfix_point[5] - unfix_point[4]
        # find diff set on z_.
        z_ = combineLineSegment(z_)
        for index in range(len(z_) - 1):
            if z_[index + 1][0] - z_[index][1] >= top_depth:
                return z_[index][1], np.abs(z_[index][1] - unfix_point[4]) > DELTA
        return unfix_point[4], False

    def checkWidth(self, unfix_point: List[float]) -> Tuple[float, bool]:
        """
        Fix item position x.
        :param unfix_point: The back-left-down and front-right-up point coordinate of the item. [x_min, x_max, y_min, y_max, z_min, z_max].
        """
        x_: List[List[float]] = [[0, 0], [self.width, self.width]]
        for fix_item_corner in self.fit_items:
            rect1 = Rectangle(left_back_x=unfix_point[2], left_back_y=unfix_point[4], right_front_x=unfix_point[3], right_front_y=unfix_point[5])
            rect2 = Rectangle(left_back_x=fix_item_corner[2], left_back_y=fix_item_corner[4], right_front_x=fix_item_corner[3], right_front_y=fix_item_corner[5])
            if checkIntersect(rect1, rect2):
                x_.append([fix_item_corner[0], fix_item_corner[1]])
        top_width = unfix_point[1] - unfix_point[0]
        # find diff set on x_bottom and x_top.
        # x_ = sorted(x_, key=lambda x_: x_[1])
        x_ = combineLineSegment(x_)
        for index in range(len(x_) - 1):
            if x_[index + 1][0] - x_[index][1] >= top_width:
                return x_[index][1], np.abs(x_[index][1] - unfix_point[0]) > DELTA
        return unfix_point[0], False

    def checkHeight(self, unfix_point: List[float]) -> Tuple[float, bool]:
        """
        Fix item position y.
        :param unfix_point: The back-left-down and front-right-up point coordinate of the item. [x_min, x_max, y_min, y_max, z_min, z_max].
        """
        y_: List[List[float]] = [[0, 0], [self.height, self.height]]
        for fix_item_corner in self.fit_items:
            rect1 = Rectangle(left_back_x=unfix_point[0], left_back_y=unfix_point[4], right_front_x=unfix_point[1], right_front_y=unfix_point[5])
            rect2 = Rectangle(left_back_x=fix_item_corner[0], left_back_y=fix_item_corner[4], right_front_x=fix_item_corner[1], right_front_y=fix_item_corner[5])
            if checkIntersect(rect1, rect2):
                y_.append([fix_item_corner[2], fix_item_corner[3]])
        item_height = unfix_point[3] - unfix_point[2]
        # find diff set on y_bottom and y_top.
        y_ = combineLineSegment(y_)
        for index in range(len(y_) - 1):
            if y_[index + 1][0] - y_[index][1] >= item_height:
                return y_[index][1], np.abs(y_[index][1] - unfix_point[2]) > DELTA

        return unfix_point[2], False

    def fixAirPlace(self, item: Item, selected_EMS: List[NDArray[np.float_]]) -> bool:
        x, y, z = selected_EMS[0]
        w, h, d = item.getDimension()
        while True:
            # fix height
            y, is_y_change = self.checkHeight([x, x + w, y, y + h, z, z + d])
            # fix width
            x, is_x_change = self.checkWidth([x, x + w, y, y + h, z, z + d])
            # fix depth
            z, is_z_change = self.checkDepth([x, x + w, y, y + h, z, z + d])
            if not (is_x_change | is_y_change | is_z_change):
                break
        item_area_lower = int(w * h)
        # Cal the surface area of the underlying support.
        support_area_upper = 0
        for item_corner in self.fit_items:
            # Verify that the lower support surface area is greater than the upper support surface area * support_surface_ratio.
            # Lower of the item to put is equal to the upper of the item putting into the bin.
            if z == item_corner[5]:
                area = (
                    len(set([j for j in range(int(x), int(x + int(w)))]) & set( [j for j in range(int(item_corner[0]), int(item_corner[1]))]))
                    * len(set([j for j in range(int(y), int(y + int(h)))]) & set( [j for j in range(int(item_corner[2]), int(item_corner[3]))]))
                )  # fmt: skip
                support_area_upper += area

        # If not , get four vertices of the bottom of the item.
        log.debug(f"Item [{item.partno}], dimension={item.getDimension()}, supported area = [{support_area_upper}], minimal supported area = [{item_area_lower * self.support_surface_ratio}].")
        if support_area_upper / item_area_lower < self.support_surface_ratio:
            four_vertices = [
                [x, y],
                [x + w, y],
                [x, y + h],
                [x + w, y + h],
            ]
            #  If any vertices is not supported, fit = False.
            c = [False, False, False, False]
            for item_corner in self.fit_items:
                if z == item_corner[5]:
                    for jdx, j in enumerate(four_vertices):
                        if (item_corner[0] <= j[0] <= item_corner[1]) and (item_corner[2] <= j[1] <= item_corner[3]):
                            c[jdx] = True
            log.debug(f"Item [{item.partno}] has [{sum(c)}] supported corners.")
            if False in c:
                return False

        self.fit_items = np.append(
            self.fit_items,
            np.array([[x, x + w, y, y + h, z, z + d]]),
            axis=0,
        )
        item.position = [x, y, z]
        return True
