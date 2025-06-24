from typing import List, Tuple

from .constants import RT_ALL, START_POSITION, RotationType


class Item:
    def __init__(
        self,
        partno: str,
        name: str,
        typeof: str,
        WHD: Tuple[float, float, float],
        weight: float,
        level: int,
        loadbear: int,
        updown: bool,
        color: str,
    ) -> None:
        self.partno: str = partno
        self.name: str = name
        self.typeof: str = typeof
        self.width: float = WHD[0]
        self.height: float = WHD[1]
        self.depth: float = WHD[2]
        self.weight: float = weight
        self.level: int = level  # Packing Priority level ,choose 1-3: The lower the number, the higher the priority.
        self.loadbear: int = loadbear  # loadbear: The higher the number, the higher the priority.
        self.updown: bool = updown if typeof == "cube" else False  # Upside down? True or False: True means the item can be placed upside down.
        self.color: str = color  # Draw item color
        self.rotation_type: RotationType = RotationType.RT_WHD
        self.position: List[float] = START_POSITION

    def string(self) -> str:
        return f"{self.partno}({self.width}x{self.height}x{self.depth}), weight={self.weight}, pos={self.position}, rotType={self.rotation_type}, volume={self.getVolume()}"

    def getVolume(self) -> float:
        return self.width * self.height * self.depth

    def getDimension(self) -> List[float]:
        """
        Get the dimension of the item according to the rotation type.
        """
        if self.rotation_type == RotationType.RT_WHD:
            dimension = [self.width, self.height, self.depth]
        elif self.rotation_type == RotationType.RT_HWD:
            dimension = [self.height, self.width, self.depth]
        elif self.rotation_type == RotationType.RT_HDW:
            dimension = [self.height, self.depth, self.width]
        elif self.rotation_type == RotationType.RT_DHW:
            dimension = [self.depth, self.height, self.width]
        elif self.rotation_type == RotationType.RT_DWH:
            dimension = [self.depth, self.width, self.height]
        elif self.rotation_type == RotationType.RT_WDH:
            dimension = [self.width, self.depth, self.height]
        else:
            raise ValueError(f"Current rotation type = {self.rotation_type}, not in {RT_ALL}")

        return dimension
