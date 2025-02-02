# from multiprocess import Pool
import datetime
import os
from typing import Union

import imageio as im
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .KernelOperations import KernelOperations as KOP

# plt.interactive(True)
# print("Interactive Mode: ", plt.isinteractive())


# TODO: Refactor
# TODO: Tracking fire shape, make algorithm to drop retardant on optimal spots
# TODO: Make a GUI for easy editability


class SimGrid(object):

    def __init__(self, time: int, forest_size: Union[tuple, list, np.array], maxfuel=5):

        self.time = int(time)  # Length of simulation
        self.forest_size = tuple(forest_size)  # Grid Size

        # Important Parameters and coefficients
        self.__r_evap = lambda r_grid, k: k / (r_grid + k)  # Base-evaporation rate curve
        self.__weights = np.array([[1 / np.sqrt(2), 1, 1 / np.sqrt(2)],
                                   [1, 0.5, 1],
                                   [1 / np.sqrt(2), 1, 1 / np.sqrt(2)]],
                                  dtype=float)  # Weights used for weighted average

        self.__retardant_efficiency = 5.0  # Lower numbers will reduce effectiveness
        self.__k = 0.001  # lower numbers will create less evaporation
        self.__gust_factor = 0  # Higher numbers will make wind more random
        self.__elevation_shift_intensity = 0  # Higher numbers will make terrain more random
        self.__maxfuel = maxfuel
        self.__retardant_droppings = {}

        # Main grid where simulation values are stored
        self.main_grid = np.zeros((self.time, *self.forest_size, 4),
                                  dtype=float)  # This grid will be used to convert to gif

        # These grids will be temporary grids used for calculations
        self.F_grid = np.random.uniform(0.05, 1, self.forest_size) * self.__maxfuel  # Fuel Grid
        self.I_grid = np.zeros(self.forest_size, dtype=float)  # Intensity Grid
        self.R_grid = np.zeros(self.forest_size, dtype=float)  # Retardant Grid
        self.D_grid = np.random.uniform(0.0, 0.1, self.forest_size)  # Fuel Dryness Index
        self.__dryness_distribution(5, 4)

        # Final coloured array
        self.__fire_rgb = [
            (128, 17, 0),
            (182, 34, 3),
            (215, 53, 2),
            (252, 100, 0),
            (255, 117, 0),
            (250, 192, 0)
        ]
        self.__ground_rgb = [
            (21, 30, 13),
            (78, 72, 36),
            (108, 128, 37),
            (107, 161, 10),
            (175, 170, 68),
            (135, 126, 61),
            (195, 197, 158)
        ]
        self.__ret_rgb = [
            (0, 71, 114),
            (0, 87, 139),
            (0, 103, 165),
            (0, 119, 190),
            (0, 135, 216),
            (0, 151, 241),
            (12, 164, 255),
            (13, 165, 255),
            (33, 172, 255),
            (72, 187, 255),
            (131, 209, 255),
            (190, 231, 255)
        ]
        self.__fire_rgb.reverse()
        self.__ground_rgb.reverse()
        self.__ret_rgb.reverse()
        self.coloured = np.zeros((self.time, *self.forest_size, 3), dtype=np.uint8)

        # Interesting parameters for statistical analysis later
        self.fuel_average = np.zeros((self.time,))
        self.retardant_average = np.zeros((self.time,))
        self.fire_average = np.zeros((self.time,))

    def clear_space(self, topleft: tuple, size: tuple):
        """
        Make part of the grid unburnable
        :param topleft: top-left coordinate of the grid
        :param size: size of the grid
        """

        self.F_grid[topleft[0]:topleft[0] + size[0], topleft[1]: topleft[1] + size[1]] = np.zeros(size, dtype=float)

    def set_fire(self, intensity: float, position: tuple):

        self.I_grid[position[0], position[1]] = intensity

    def place_retardant(self, amount: float, topleft: tuple, size: tuple, randomness: float = 0):
        """
        Set a certain part of the grid as retardant
        :param amount: Amount of retardant, 0-1
        :param topleft: top-left coordinate of the grid
        :param size: size of the grid
        :param randomness: evenness of retardant distribution. 0 produces a fully uniform distribution, 1 a fully random
        """

        retardant = np.ones(size) * amount

        if randomness == 0:
            pass

        else:
            delta_distribution = np.random.uniform(-amount, amount, size) * randomness
            retardant += delta_distribution
            retardant[retardant > 1] = 1
            retardant[retardant < 0] = 0

        self.R_grid[topleft[0]:topleft[0] + size[0], topleft[1]: topleft[1] + size[1]] = retardant

    def set_retardant_efficiency(self, e: Union[float, int]):
        """
        Higher numbers will make retardant slower to evaporate with fire
        :param e: efficiency; 0 = useless, normal setting: 2.5-7.5
        """

        self.__retardant_efficiency = abs(e)

    def set_retardant_evaporation_constant(self, k: float = 0.005):
        """
        Higher numbers will make retardant evaporate faster
        :param k: normal values: 0.0001-0.005
        """

        self.__k = k

    def set_wind(self, direction: Union[tuple, list, np.array, np.ndarray], magnitude: Union[float, int],
                 randomness: float = 0.2):
        """
        Set the general wind direction for the simulation
        :param direction: (steps in x ('downwards'), steps in y ('right'))
        :param magnitude: 0-1
        :param randomness: 0-1
        """
        if len(direction) == 2:
            direction = np.array(direction, dtype=float)
            direction = direction / np.linalg.norm(direction)
        else:
            raise ValueError("Wrong Shape input for direction")

        if 1 >= magnitude >= 0:
            pass
        else:
            raise ValueError("Magnitude has to be between 1 and 0")

        self.__weights += magnitude * self.__directed_weights_increase(direction)
        self.__gust_factor = randomness
        # self.__weights += np.random.uniform(-self.__gust_factor, self.__gust_factor, self.__weights.shape)

    def set_elevation(self, slope_direction: Union[tuple, list, np.array, np.ndarray], slope_angle: float,
                      randomness: float = 0.2):
        """
        Set the elevation-map of the grid
        :param slope_direction: (steps in x ('downwards'), steps in y ('right'))
        :param slope_angle: 0 - pi/2
        :param randomness: 0-1
        """

        if len(slope_direction) == 2:
            direction = np.array(slope_direction, dtype=float)
            direction = direction / np.linalg.norm(direction)
        else:
            raise ValueError("Wrong Shape input for direction")

        if np.pi / 2 > slope_angle >= 0:
            pass

        else:
            raise ValueError("Magnitude has to be between 1 and 0")

        self.__weights += slope_angle / np.pi * self.__directed_weights_increase(direction)
        self.__elevation_shift_intensity = randomness

    def run(self, name: str = None, path: str = None):
        """
        Start the simulation
        :param name: name of file
        :param path: folder of file
        """

        if path is None:
            path = '\\'.join(os.getcwd().split('\\')) + '\\simulations\\'

        if name is None:
            time = datetime.datetime.now()
            date = time.date()
            date = f"{date.month}_{date.day}"
            time = time.time()
            time = f"{time.hour}_{time.minute}_{time.second}"
            name = f"sim_{date}_{time}"

        print("===== SIMULATING =====")
        for t in tqdm(range(self.time)):
            self.fuel_average[t] = KOP.grid_average(self.F_grid)
            self.fire_average[t] = KOP.grid_average(self.I_grid)
            self.retardant_average[t] = KOP.grid_average(self.R_grid)
            # print(self.fire_average[t], self.fuel_average[t])
            self.__update(t)

        print("===== SAVING =====")
        self.__save_simulation(name, path)
        print("===== DONE =====")

    def plot_progress(self):
        """
        Plot the averages of the 3 parameters over time
        """
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(range(self.time), self.fuel_average, 'g--', label='Fuel Amount Average')
        ax[1].plot(range(self.time), self.fire_average, 'r--', label='Fire Intensity Average')
        ax[2].plot(range(self.time), self.retardant_average, 'b--', label='Retardant Amount Average')

        for i in range(3):

            if i == 0:
                ax[i].set_ylim(0, np.max(self.fuel_average))
                ax[i].set_title('Average Fuel - Time')
                ax[i].set_ylabel('Average Fuel Amount')

            elif i == 1:
                ax[i].set_ylim(0, np.max(self.fire_average))
                ax[i].set_title('Average Fire Intensity - Time')
                ax[i].set_ylabel('Average Fire Intensity')

            elif i == 2:
                ax[i].set_ylim(0, np.max(self.retardant_average))
                ax[i].set_title('Average Retardant - Time')
                ax[i].set_ylabel('Average Retardant Amount')

            ax[i].set_xlim(0, self.time)
            ax[i].grid(True)
            ax[i].set_xlabel('Time')

        fig.tight_layout()
        # plt.show()

    def set_dry_patch(self, top_left: tuple, bottom_right: tuple, amount: float, randomness=0.3):

        for x in range(top_left[0], bottom_right[0]+1):
            for y in range(top_left[1], bottom_right[1]+1):
                self.D_grid[x, y] = amount + np.random.uniform(-randomness, randomness)

    def __dryness_distribution(self, n_strips: int, randomness: int = 5):
        """

        :param n_strips:
        :param randomness:
        :return:
        """

        for n in range(n_strips):

            xs = np.random.randint(0, self.D_grid.shape[0], size=2)
            ys = np.random.randint(0, self.D_grid.shape[1], size=2)

            p_0 = np.array([xs[0], ys[0]])
            p_1 = np.array([xs[1], ys[1]])

            idcs = [list(i) for i in KOP.intersection_point_finder(self.D_grid, p_0, p_1)]

            for i in range(len(idcs)):
                idcs[i][0] += np.random.randint(-randomness, randomness)
                idcs[i][1] += np.random.randint(-randomness, randomness)

            for x, y in idcs:

                if x < 0 or x >= self.D_grid.shape[0] or y < 0 or y >= self.D_grid.shape[1]:
                    continue

                values, idc, result_shape = KOP.extract_kernel(self.D_grid, (x, y), shape=(3, 3), return_info=True)

                self.D_grid[idc[0][0]:idc[0][1] + 1, idc[1][0]:idc[1][1] + 1] = values + np.random.uniform(0.0, 0.2, result_shape)

    @staticmethod
    def __directed_weights_increase(direction: Union[np.array, np.ndarray]):
        """
        Calculate the delta in the weights for the kernel
        :param direction: 2 by 1 vector
        :return: delta weights matrix
        """

        weights = np.zeros((3, 3))

        for x in range(3):
            for y in range(3):
                weights[x, y] = -direction[0] * (x - 1) - direction[1] * (y - 1)

        return weights

    def __initialize_grid(self):
        """
        Implements initial values into main grid
        """

        # Set up main grid
        self.main_grid[0, :, :, 0] = np.array(self.F_grid)
        self.main_grid[0, :, :, 1] = np.array(self.I_grid)
        self.main_grid[0, :, :, 2] = np.array(self.R_grid)
        self.main_grid[0, :, :, 3] = np.array(self.D_grid)

    def __kernel_average(self, grid: Union[np.array, np.ndarray], position: tuple, base_weights: bool = True,
                         directional_weights: bool = True):
        """
        Method to calculate a weighted average of a 3x3 kernel
        :param grid: grid upon which the calculation must be performed
        :param position:
        :return:
        """

        size = [3, 3]
        weight_idx = [[0, 3], [0, 3]]
        xrange = [position[0] - 1, position[0] + 1]
        yrange = [position[1] - 1, position[1] + 1]

        if xrange[0] < 0:
            xrange[0] = 0
            weight_idx[0][0] += 1
            size[0] -= 1

        elif xrange[1] >= grid.shape[0]:
            xrange[1] = grid.shape[0] - 1
            weight_idx[0][1] -= 1
            size[0] -= 1

        if yrange[0] < 0:
            yrange[0] = 0
            weight_idx[1][0] += 1
            size[1] -= 1

        elif yrange[1] >= grid.shape[1]:
            yrange[1] = grid.shape[1] - 1
            weight_idx[1][1] -= 1
            size[1] -= 1

        if directional_weights and base_weights:
            new_weights = self.__weights[weight_idx[0][0]:weight_idx[0][1], weight_idx[1][0]:weight_idx[1][1]] + \
                          KOP.random_kernel(tuple(size),
                                               (-self.__gust_factor, self.__gust_factor)) + \
                          KOP.random_kernel(tuple(size),
                                               (-self.__elevation_shift_intensity, self.__elevation_shift_intensity))

        elif base_weights and not directional_weights:
            new_weights = self.__weights[weight_idx[0][0]:weight_idx[0][1], weight_idx[1][0]:weight_idx[1][1]]

        else:
            new_weights = np.ones(shape=tuple(size))

        return np.average(grid[xrange[0]:xrange[1] + 1, yrange[0]:yrange[1] + 1], weights=new_weights)

    def retardant_along_line(self, t: int,
                             amount: float,
                             p_0: Union[np.array, np.ndarray],
                             p_1:  Union[np.array, np.ndarray],
                             velocity: float,
                             shape: tuple = (3, 3),
                             randomness: float = 0.2):
        """

        :param t:
        :param amount:
        :param length:
        :param p_0:
        :param direction:
        :param shape:
        :param randomness:
        :return:
        """

        p_1 = np.array(p_1)
        p_0 = np.array(p_0)
        direction = p_1 - p_0
        length = np.linalg.norm(direction)

        indices = KOP.intersection_point_finder(self.main_grid[t, :, :, 2], p_0, p_1)
        kernel_indices = KOP.get_kernel_indices(indices, shape)

        time = int(np.ceil(length/velocity))
        time_chunk = len(kernel_indices)//time+1
        result = np.zeros(self.forest_size, dtype=float)

        for idx, ti in enumerate(range(t, t+time)):

            for coordinates in kernel_indices[time_chunk*idx:time_chunk*(idx+1)]:

                for x, y in coordinates:

                    try:
                        result[x, y] += np.random.uniform(0, 1) * amount

                    except IndexError:
                        continue

            if str(ti) not in self.__retardant_droppings.keys():
                self.__retardant_droppings[str(ti)] = [result, ]

            else:
                self.__retardant_droppings[str(ti)].append(result)

            result = np.zeros(self.forest_size, dtype=float)

    def __intensity_averages(self):
        """
        Averaging the fire intensity over a 3x3 kernel,every grid with their own weights
        :return:
        """

        avgs = np.zeros(self.forest_size, dtype=float)

        for x in range(self.forest_size[0]):
            for y in range(self.forest_size[1]):
                avgs[x, y] = self.__kernel_average(self.I_grid, (x, y))

        return avgs

    def __update_fuel(self, t: int):
        """

        :param t:
        :return:
        """

        old_grid = np.array(self.F_grid)

        for x in range(self.forest_size[0]):
            for y in range(self.forest_size[1]):
                if self.R_grid[x, y] > 0:
                    if self.R_grid[x, y] >= self.I_grid[x, y]:
                        self.F_grid[x, y] = old_grid[x, y]

                    else:
                        self.F_grid[x, y] = old_grid[x, y] + (self.R_grid[x, y] - self.I_grid[x, y])*(1 + self.D_grid[x, y])

                else:
                    self.F_grid[x, y] = old_grid[x, y] - self.I_grid[x, y]*(1 + self.D_grid[x, y])

        self.F_grid[self.F_grid < 0.0001] = 0
        # self.F_grid[self.F_grid > 1] = 1
        self.main_grid[t, :, :, 0] = self.F_grid

        return old_grid

    def __update_retardant(self, t: int):
        """

        :param t:
        :return:
        """

        old_grid = np.array(self.R_grid)
        # self.R_grid = (1 - (self.__r_evap(self.R_grid)+self.I_grid))*self.R_grid

        evap = self.__r_evap(old_grid, k=self.__k)

        for x in range(self.forest_size[0]):
            for y in range(self.forest_size[1]):
                if self.R_grid[x, y] >= self.I_grid[x, y]:
                    self.R_grid[x, y] = (1 - (evap[x, y] + self.__kernel_average(self.I_grid, (x, y),
                                                                                 base_weights=False,
                                                                                 directional_weights=False))) * \
                                        self.R_grid[x, y]

                else:
                    self.R_grid[x, y] = 0

        if str(t) in self.__retardant_droppings.keys():

            for value in self.__retardant_droppings[str(t)]:
                self.R_grid += value

        self.R_grid[self.R_grid < 0.0001] = 0
        # self.R_grid[self.R_grid > 1] = 1
        self.main_grid[t, :, :, 2] = self.R_grid
        return old_grid

    def __update_intensities(self, t: int,
                             old_fuel: Union[np.array, np.ndarray],
                             old_retardant: Union[np.array, np.ndarray]):
        """

        :param t:
        :param old_fuel:
        :param old_retardant:
        :return:
        """

        old_grid = np.array(self.I_grid)

        delta_fuel = np.array(self.F_grid - old_fuel)
        delta_retardant = np.array(self.R_grid - old_retardant)

        iavgs = KOP.generate_averaged_grid(self.I_grid, self.__weights)

        # print(iavgs[self.forest_size[0] // 2 - 1 : self.forest_size[0] // 2 + 2, self.forest_size[1] // 2 - 1: self.forest_size[1] // 2 + 2])

        for x in range(self.forest_size[0]):
            for y in range(self.forest_size[1]):

                if old_fuel[x, y] > 0:
                    if self.I_grid[x, y] > 0:
                        self.I_grid[x, y] = old_grid[x, y] + iavgs[x, y]/(1+self.__retardant_efficiency) + \
                                            abs(delta_fuel[x, y]) + \
                                            delta_retardant[x, y] * self.__retardant_efficiency

                    else:
                        # if self.R_grid[x, y] > 0:
                        self.I_grid[x, y] = (old_grid[x, y] + iavgs[x, y] - old_retardant[x, y])

                else:
                    if self.I_grid[x, y] > 0:
                        self.I_grid[x, y] = (old_grid[x, y] + iavgs[x, y]) / 10
                    else:
                        self.I_grid[x, y] = 0

        self.I_grid[self.I_grid < 0.0001] = 0
        self.I_grid[self.I_grid > 1] = 1
        self.main_grid[t, :, :, 1] = self.I_grid
        return old_grid

    def __update(self, t: int):

        if t == 0:
            self.__initialize_grid()
        else:
            old_fuel = self.__update_fuel(t)
            old_retardant = self.__update_retardant(t)
            old_intensity = self.__update_intensities(t, old_fuel, old_retardant)

        self.__create_coloured_grid(t)

    def __create_coloured_grid(self, t: int):

        for x in range(self.forest_size[0]):
            for y in range(self.forest_size[1]):
                cell = self.main_grid[t, x, y, :]

                if cell[1] > 0 and cell[2] > 0:
                    if cell[1] > cell[2]:
                        try:
                            colour = self.__fire_rgb[int(np.round(cell[1] * len(self.__fire_rgb), 0))]

                        except IndexError:
                            colour = self.__fire_rgb[-1]

                    else:
                        try:
                            colour = self.__ret_rgb[int(np.round(cell[2] * len(self.__ret_rgb), 0))]

                        except IndexError:
                            colour = self.__ret_rgb[-1]

                elif cell[1] > 0 and cell[2] == 0:
                    try:
                        colour = self.__fire_rgb[int(np.round(cell[1] * len(self.__fire_rgb), 0))]

                    except IndexError:
                        colour = self.__fire_rgb[-1]

                elif cell[1] == 0 and cell[2] > 0:
                    try:
                        colour = self.__ret_rgb[int(np.round(cell[2] * len(self.__ret_rgb), 0))]

                    except IndexError:
                        colour = self.__ret_rgb[-1]

                elif cell[1] == 0 and cell[2] == 0:
                    if cell[0] == 0:
                        colour = self.__ground_rgb[-1]

                    else:
                        try:
                            colour = self.__ground_rgb[
                                int(np.round(cell[0] / self.__maxfuel * len(self.__ground_rgb), 0))]

                        except IndexError:
                            colour = self.__ground_rgb[-1]

                else:
                    raise ValueError(f"Could not assign colour to grid point {(x, y)} with {[i for i in cell]}")

                self.coloured[t, x, y, :] = colour

    def __save_simulation(self, name: str, path: str, output: str = 'mp4'):
        """
        Save simulation as gif
        :param name: name of file
        :param path: path to older in which to store
        """

        im.mimsave((path + f'/{name}.{output}'), self.coloured, fps=10, macro_block_size=10)



