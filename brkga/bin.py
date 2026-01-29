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
        self.partno: str = partno
        self.items: List[Item] = []
        self.width: float = WHD[0]
        self.height: float = WHD[1]
        self.depth: float = WHD[2]
        self.dimensions: Tuple[float, float, float] = WHD
        self.EMSs: NDArray[np.float64] = np.array([[0, 0, 0, WHD[0], WHD[1], WHD[2]]])  # The ems (x, y, z, x+w, y+h, z+d) of the left-back-down and right-front-up corners of the empty maximal space.
        self.fit_items: NDArray[np.float64] = np.array([[0, WHD[0], 0, WHD[1], 0, 0]])  # The ems (x, x+w, y, y+h, z, z+d) of the left-back-down and right-front-up corners of the items placed in the bin.
        assert 0 < support_surface_ratio <= 1, f"Should be in [0, 1]."
        self.support_surface_ratio = support_surface_ratio
        log.info(f"Initial EMSs:\n{self.getEMSs()}")

    def getVolume(self) -> float:
        """
        Compute the volume of the bin.
        """
        return self.width * self.height * self.depth

    def updateEMS(self, item: Item, min_vol: float, min_dim: float, is_debug: bool = False) -> None:
        """
        Update the remaining EMSs of the bin according to the new placed item.
        :param item: The new placed item.
        :param min_vol: Minimal volume of all the unplaced items.
        :param min_dim: Minimal dimension of all the unplaced items.
        :param is_debug: Whether to plot the placed item and the generated EMS for debug.
        """
        # Place the item into the bin.
        x, y, z = item.position  # Should not use the `selected_EMS` because we modify the position of the item.
        w, h, d = item.getDimension()
        self.items.append(item)
        item_ems: NDArray[np.float64] = np.array([x, y, z, x + w, y + h, z + d])
        log.info(f"EMS of the placed items:\n{item_ems.tolist()}, size: {item.getDimension()}")

        # Eliminate the remaining EMSs by check whether the EMS is inscribed or overlapped with the current item.
        inscribed_mask = self.checkInscribed(self.EMSs, np.array([item_ems]))
        log.debug(f"Remove inscribed EMSs:\n{self.EMSs[inscribed_mask].tolist()}")
        self.EMSs = self.EMSs[np.logical_not(inscribed_mask)]
        overlapped_mask = self.checkOverlapped(item_ems, self.EMSs)
        overlapped_EMSs = deepcopy(self.EMSs[overlapped_mask])
        log.debug(f"Remove overlapped EMSs:\n{overlapped_EMSs.tolist()}")
        self.EMSs = self.EMSs[np.logical_not(overlapped_mask)]
        log.info(f"Remaining EMSs after removing the inscribed and overlapped EMSs:\n{self.getEMSs()}")

        # Generate new EMSs from the intersection of the item and the overlapped EMSs.
        for overlapped_EMS in overlapped_EMSs:
            # Add 6 new EMSs in 3 dimensions.
            x1, y1, z1, x2, y2, z2 = overlapped_EMS
            x3, y3, z3, x4, y4, z4 = item_ems
            new_EMSs = np.vstack(
                [
                    np.array([x4, y1, z1, x2, y2, z2]),  # front EMS
                    np.array([x1, y4, z1, x2, y2, z2]),  # right EMS
                    np.array([x1, y1, z4, x2, y2, z2]),  # up EMS
                    np.array([x1, y1, z1, x3, y2, z2]),  # back EMS
                    np.array([x1, y1, z1, x2, y3, z2]),  # left EMS
                    np.array([x1, y1, z1, x2, y2, z3]),  # down EMS
                ]
            )
            new_EMSs_size = new_EMSs[:, 3:] - new_EMSs[:, :3]
            # Use `DELTA` to avoid the volume or the dimension is equal to zero when placing the last item.
            big_vol_EMS_index = np.prod(new_EMSs_size, axis=1) >= (min_vol if min_vol > 0 else DELTA)  # Do not add new EMS smaller than the volume of remaining items.
            big_dim_EMS_index = np.min(new_EMSs_size, axis=1) >= (min_dim if min_dim > 0 else DELTA)  # Do not add new EMS whose smallest dimension is smaller than the smallest dimension of remaining items.
            log.debug(f"Small volume EMS:\n{new_EMSs[np.logical_not(big_vol_EMS_index)]}")
            log.debug(f"Small dimension EMS:\n{new_EMSs[np.logical_not(big_dim_EMS_index)]}")
            index = big_vol_EMS_index & big_dim_EMS_index
            new_EMSs = new_EMSs[index]
            assert np.all(new_EMSs_size[index] > 0), f"All size should be larger than 0: {new_EMSs_size[index].tolist()}."

            if new_EMSs.size > 0:
                # Eliminate new EMSs which are totally inscribed by the remaining EMSs.
                log.debug(f"New available EMSs candidates:\n{new_EMSs.tolist()}")
                new_EMSs_index = np.logical_not(self.checkInscribed(new_EMSs, self.EMSs))
                new_EMSs = new_EMSs[new_EMSs_index]
                if new_EMSs.size > 0:
                    remaining_EMS_index = np.logical_not(self.checkInscribed(self.EMSs, new_EMSs))
                    self.EMSs = self.EMSs[remaining_EMS_index]
                    if self.EMSs.size > 0:
                        self.EMSs = np.vstack([self.EMSs, new_EMSs])
                    else:
                        self.EMSs = new_EMSs
                    log.info(f"Successfully added new EMS:\n{new_EMSs.tolist()}")

                if is_debug:
                    from .painter import Painter

                    # For debug, all ems in the `self.EMss` should not overlap with each other.
                    for index, curr_EMS in enumerate(self.EMSs):
                        left_EMSs = np.vstack([self.EMSs[:index], self.EMSs[index + 1 :]])
                        mask = self.checkInscribed(np.array([curr_EMS]), left_EMSs)
                        assert not np.any(mask), f"{curr_EMS=}, touch with the existing EMS:\n{self.EMSs.tolist()}"
                    # painter = Painter(self)
                    # plt = painter.plotItemsAndBin()
                    # plt = painter.plotRemainedEMS()
                    # plt.show()

        log.warning(f"Remaining {len(self.EMSs)} EMSs:\n{self.getEMSs()}")

    @staticmethod
    def checkOverlapped(item_ems: NDArray[np.float64], remaining_EMS: NDArray[np.float64]) -> NDArray[np.bool_]:
        """
        Check whether the remaining EMSs of the bin is overlapped with the placed item.
        :param item_ems: The ems of the placed item.
        :param remaining_EMS: The remaining EMSs of the bin.
        """
        row, _ = remaining_EMS.shape
        # assert row > 0, f"Row should larger than 0, now [{row = }]."  # FIXME: when the `remaining_EMS` is an empty array, will return an empty array.
        item_ems_repeat = np.tile(item_ems, (row, 1))
        # Exclude the "=", otherwise two ems would be considered as overlapped when they have one side in contact with each other.
        mask: NDArray[np.bool_] = np.all(item_ems_repeat[:, 3:] > remaining_EMS[:, :3], axis=1) & np.all(item_ems_repeat[:, :3] < remaining_EMS[:, 3:], axis=1)
        return mask

    @staticmethod
    def checkInscribed(ems1: NDArray[np.float64], ems2: NDArray[np.float64]) -> NDArray[np.bool_]:
        """
        Check whether ems1 is inscribed by ems2. The order of the parameters is important.
        :param ems1: The first EMS.
        :param ems2: The second EMS.
        """
        row1, _ = ems1.shape
        row2, _ = ems2.shape
        if row1 == 0:
            return np.array([], dtype=bool)
        if row2 == 0:
            return np.array([False] * row1, dtype=bool)
        # assert row1 * row2 > 0, f"All shape should larger than 0: {row1 = }, {row2 = }."
        ems1_repeat = np.repeat(ems1, row2, axis=0).reshape(row1, row2, -1)
        ems2_repeat = np.tile(ems2, (row1, 1, 1))
        # Assume row1=2, row2=3
        # ems1_repeat = [ems1[1,:],ems1[1,:],ems1[1,:];ems1[2,:],ems1[2,;],ems1[2,:]]
        # ems2_repeat = [ems2[1,:],ems2[2,:],ems2[3,:];ems2[1,:],ems2[2,;],ems2[3,:]]
        diff = ems2_repeat - ems1_repeat
        mask: NDArray[np.bool_] = np.any((np.all(diff[:, :, :3] <= 0, axis=2) & np.all(diff[:, :, 3:] >= 0, axis=2)), axis=1)
        return mask

    def getEMSs(self) -> List[np.float64]:
        """
        Return the remaining EMSs of the bin.
        """
        remaining_EMSs_list: List[np.float64] = self.EMSs.tolist()
        return remaining_EMSs_list

    def calculateUtilizationRate(self) -> float:
        """
        Calculate the utilization rate of the bin.
        """
        return float(np.sum([np.prod(item[1::2] - item[::2]) for item in self.fit_items]) / np.prod(np.array(self.dimensions), axis=0))

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

    def fixAirPlace(self, item: Item, selected_EMS: NDArray[np.float64]) -> bool:
        """
        Fix the item in the air and check the stability of the placed item.
        :param item: The item to place.
        :param selected_EMS: The EMS whose origin the item is placed into at first.
        """
        x, y, z = selected_EMS[:3]
        log.debug(f"Item position before fix: {selected_EMS[:3].tolist()}.")
        w, h, d = item.getDimension()
        while True:
            # fix height
            y, is_y_change = self.checkHeight([x, x + w, y, y + h, z, z + d])
            # fix width
            x, is_x_change = self.checkWidth([x, x + w, y, y + h, z, z + d])
            # fix depth
            z, is_z_change = self.checkDepth([x, x + w, y, y + h, z, z + d])
            if not (is_x_change | is_y_change | is_z_change):
                log.debug(f"The position of the item will not be changed.")
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
        log.debug(f"Item position after fix: {item.position}.")
        return True
