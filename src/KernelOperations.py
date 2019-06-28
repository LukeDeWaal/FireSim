import numpy as np
from typing import Union


class KernelOperations:

    @staticmethod
    def random_kernel(shape: tuple, randrange: tuple = (-0.2, 0.2)):
        return np.random.uniform(*randrange, shape)

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
        """
        Weighted average of a grid
        :param grid: Grid to calculate average of
        :param weights: Matrix of weights, same shape as grid
        :return:
        """
        if weights.shape != grid.shape:
            print(weights, grid)
            raise IndexError("Shapes don't match")

        return np.average(grid, weights=weights)

    @staticmethod
    def extract_kernel(grid: Union[np.array, np.ndarray], position: tuple, shape: tuple, return_info: bool = False):
        """
        Extract a sub-grid from grid
        :param grid: Main grid
        :param position: Position of centre of sub grid
        :param shape: Shape of subgrid
        :return: Subgrid
        """
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

            if return_info is True:
                return kernel, grid_idx, size

            elif return_info is False:
                return kernel

        else:

            raise IndexError("Coordinate outside grid")

    @classmethod
    def generate_averaged_grid(cls, grid: Union[np.array, np.ndarray], weights: Union[np.array, np.ndarray]):
        """
        Calculate (weighted) kernel averages for every cell in the grid
        :param grid: Grid to take cells from
        :param weights: Weights for calculations
        :param kernel_shape: Shape of the kernel
        :return: Grid with averages
        """
        averaged_grid = np.array(grid.shape, dtype=float)
        kernel_shape = weights.shape

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):

                kernel = cls.extract_kernel(grid, (i, j), shape=kernel_shape, return_info=False)
                averaged_grid[i, j] = cls.weighted_grid_average(kernel, weights)

        return averaged_grid

    @staticmethod
    def intersection_point_finder(grid: Union[np.array, np.ndarray],
                                  p_0: Union[np.array, np.ndarray],
                                  p_1: Union[np.array, np.ndarray],
                                  max_iter: int = 200):

        """
        Function to calculate all cells the line intersects through linear inerpolation
        :param grid: Grid in which the interpolation has to be performed
        :param p_0: starting position
        :param p_1: end position
        :param max_iter: maximum amount of samples
        :return: [(i0, j0), (i1, j1], ..., (in, jn)
        """

        indices = []

        length = np.linalg.norm(p_1 - p_0)
        v = (p_1 - p_0) / length

        for n in np.linspace(0, length, max_iter):
            p_i = p_0 + n * v

            if 0 <= p_i[0] <= grid.shape[0] and 0 <= p_i[1] <= grid.shape[1]:
                rounded = p_i.round()
                val = tuple(rounded.astype(int))

                if val not in indices:
                    indices.append(val)

            else:
                break

        return indices

    @staticmethod
    def get_kernel_indices(idcs: list, shape: tuple = (3, 3)):

        duplicate_check = set()
        new_idcs = []

        for i, j in idcs:
            coordinates = []
            for di in range(-shape[0] // 2, shape[0] // 2 + 1):
                for dj in range(-shape[1] // 2, shape[1] // 2 + 1):
                    idx = (i + di, j + dj)

                    if idx not in duplicate_check:
                        duplicate_check.add(idx)
                        coordinates.append(idx)
            new_idcs.append(coordinates)

        return new_idcs