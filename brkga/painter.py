from typing import Any, Optional

import matplotlib.pyplot as plt  # type: ignore[import]
import mpl_toolkits.mplot3d.art3d as art3d  # type: ignore[import]
import numpy as np
from matplotlib.axes import Axes  # type: ignore[import]
from matplotlib.patches import Circle, Rectangle  # type: ignore[import]

from .bin import Bin


class Painter:
    def __init__(self, bins: Bin) -> None:
        self.items = bins.items
        self.width = bins.width
        self.height = bins.height
        self.depth = bins.depth
        self.remainedEMSs = bins.EMSs

    @staticmethod
    def _plotCube(
        ax: Axes.axes,
        x: float,
        y: float,
        z: float,
        dx: float,
        dy: float,
        dz: float,
        text: Optional[str] = None,
        color: str = "red",
        mode: int = 2,
        linewidth: int = 1,
        fontsize: int = 15,
        alpha: float = 0.5,
    ) -> None:
        """
        Auxiliary function to plot a cube. code taken somewhere from the web.
        :param mode: 1 means plotting the in while 2 means plotting the rectangle item.
        """
        xx = [x, x, x + dx, x + dx, x]
        yy = [y, y + dy, y + dy, y, y]

        kwargs = {"alpha": 1, "color": color, "linewidth": linewidth}
        if mode == 1:
            ax.plot3D(xx, yy, [z] * 5, **kwargs)
            ax.plot3D(xx, yy, [z + dz] * 5, **kwargs)
            ax.plot3D([x, x], [y, y], [z, z + dz], **kwargs)
            ax.plot3D([x, x], [y + dy, y + dy], [z, z + dz], **kwargs)
            ax.plot3D([x + dx, x + dx], [y + dy, y + dy], [z, z + dz], **kwargs)
            ax.plot3D([x + dx, x + dx], [y, y], [z, z + dz], **kwargs)
        else:
            p = Rectangle((x, y), dx, dy, fc=color, ec="black", alpha=alpha)
            p2 = Rectangle((x, y), dx, dy, fc=color, ec="black", alpha=alpha)
            p3 = Rectangle((y, z), dy, dz, fc=color, ec="black", alpha=alpha)
            p4 = Rectangle((y, z), dy, dz, fc=color, ec="black", alpha=alpha)
            p5 = Rectangle((x, z), dx, dz, fc=color, ec="black", alpha=alpha)
            p6 = Rectangle((x, z), dx, dz, fc=color, ec="black", alpha=alpha)
            ax.add_patch(p)
            ax.add_patch(p2)
            ax.add_patch(p3)
            ax.add_patch(p4)
            ax.add_patch(p5)
            ax.add_patch(p6)

            if text:
                ax.text((x + dx / 2), (y + dy / 2), (z + dz / 2), str(text), color="black", fontsize=fontsize, ha="center", va="center")

            art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
            art3d.pathpatch_2d_to_3d(p2, z=z + dz, zdir="z")
            art3d.pathpatch_2d_to_3d(p3, z=x, zdir="x")
            art3d.pathpatch_2d_to_3d(p4, z=x + dx, zdir="x")
            art3d.pathpatch_2d_to_3d(p5, z=y, zdir="y")
            art3d.pathpatch_2d_to_3d(p6, z=y + dy, zdir="y")

    @staticmethod
    def _plotCylinder(
        ax: Axes.axes,
        x: float,
        y: float,
        z: float,
        dx: float,
        dy: float,
        dz: float,
        text: Optional[str] = None,
        color: str = "red",
        fontsize: int = 10,
        alpha: float = 0.2,
    ) -> None:
        """
        Auxiliary function to plot a Cylinder.
        """

        # Plot the two circles above and below the cylinder.
        p = Circle((x + dx / 2, y + dy / 2), radius=dx / 2, color=color, alpha=0.5)
        p2 = Circle((x + dx / 2, y + dy / 2), radius=dx / 2, color=color, alpha=0.5)
        ax.add_patch(p)
        ax.add_patch(p2)
        art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
        art3d.pathpatch_2d_to_3d(p2, z=z + dz, zdir="z")

        # Plot a circle in the middle of the cylinder.
        center_z = np.linspace(0, dz, 10)
        theta = np.linspace(0, 2 * np.pi, 10)
        theta_grid, z_grid = np.meshgrid(theta, center_z)
        x_grid = dx / 2 * np.cos(theta_grid) + x + dx / 2
        y_grid = dy / 2 * np.sin(theta_grid) + y + dy / 2
        z_grid = z_grid + z
        ax.plot_surface(x_grid, y_grid, z_grid, shade=False, fc=color, alpha=alpha, color=color)
        if text:
            ax.text((x + dx / 2), (y + dy / 2), (z + dz / 2), str(text), color="black", fontsize=fontsize, ha="center", va="center")

    def plotItemsAndBin(self, title: str = "", alpha: float = 0.2, write_num: bool = False, fontsize: int = 10) -> Any:
        """
        Side effective. Plot the bin and the items it contains.
        """
        fig = plt.figure()
        axGlob = plt.axes(projection="3d")

        # Plot bin.
        self._plotCube(axGlob, 0, 0, 0, self.width, self.height, self.depth, color="black", mode=1, linewidth=2)

        # Fit rotation type.
        counter = 0
        for item in self.items:
            x, y, z = item.position
            [w, h, d] = item.getDimension()
            color = item.color
            text = item.partno if write_num else None

            if item.typeof == "cube":
                # Plot item of cube.
                self._plotCube(axGlob, x, y, z, w, h, d, color=color, mode=2, text=text, fontsize=fontsize, alpha=alpha)
            elif item.typeof == "cylinder":
                # Plot item of cylinder.
                self._plotCylinder(axGlob, x, y, z, w, h, d, color=color, text=text, fontsize=fontsize, alpha=alpha)

            counter = counter + 1

        plt.title(title)
        self.setAxesEqual(axGlob)
        return plt

    def plotRemainedEMS(self, title: str = "Remain EMS", alpha: float = 0.2, fontsize: int = 10) -> Any:
        """
        Side effective. Plot the bin and the items it contains.
        """
        fig = plt.figure()
        axGlob = plt.axes(projection="3d")

        # Plot bin.
        self._plotCube(axGlob, 0, 0, 0, self.width, self.height, self.depth, color="black", mode=1, linewidth=2)

        for ems in self.remainedEMSs:
            x, y, z = ems[0]
            [w, h, d] = ems[1] - ems[0]
            self._plotCube(axGlob, x, y, z, w, h, d, color="red", mode=2, fontsize=fontsize, alpha=alpha)

        plt.title(title)
        self.setAxesEqual(axGlob)
        return plt

    @staticmethod
    def setAxesEqual(ax: Axes.axes) -> None:
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres, cubes as cubes, etc.
        This is one possible solution to Matplotlib's ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
