import numpy as np
from typing import Union


class KernelOperations:

    @staticmethod
    def grid_average(grid: Union[np.array, np.ndarray]) -> float:
        """
        Average value of a n x m shaped grid
        :param grid: Grid to calculate the average of
        :return: Average value (float)
        """
        return np.average(grid)

    @staticmethod
    def weighted_grid_average(grid: Union[np.array, np.ndarray], weights: Union[np.array, np.ndarray]) -> float:

        if weights.shape != grid.shape:
            raise IndexError("Shapes don't match")

        return np.average(grid, weights=weights)

    @staticmethod
    def extract_kernel(grid: Union[np.array, np.ndarray], position: tuple, shape: tuple):

        size = list(shape)

        if grid.shape[0] >= position[0] >= 0 and grid.shape[1] >= position[1] >= 0:

            grid_idx = [[position[0] - size[0] // 2, position[0] + size[0] // 2],
                        [position[1] - size[1] // 2, position[1] + size[1] // 2]]

            if grid_idx[0][0] < 0:
                grid_idx[0][0] = 0

            elif grid_idx[0][1] >= grid.shape[0]:
                grid_idx[0][1] = grid.shape[0] - 1

            size[0] = grid_idx[0][1] - grid_idx[0][0] + 1

            if grid_idx[1][0] < 0:
                grid_idx[1][0] = 0

            elif grid_idx[1][1] >= grid.shape[1]:
                grid_idx[1][1] = grid.shape[1] - 1

            size[1] = grid_idx[1][1] - grid_idx[1][0] + 1

            kernel = grid[grid_idx[0][0]:grid_idx[0][1] + 1, grid_idx[1][0]:grid_idx[1][1] + 1]

            return kernel

        else:

            raise IndexError("Coordinate outside grid")

    @classmethod
    def generate_averaged_grid(cls, grid: Union[np.array, np.ndarray], weights: Union[np.array, np.ndarray], kernel_shape: tuple = (3, 3)):

        averaged_grid = np.array(grid.shape, dtype=float)

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):

                kernel = cls.extract_kernel(grid, (i, j), shape=kernel_shape)

                averaged_grid[i, j] = cls.weighted_grid_average(kernel, weights)

        return averaged_grid

