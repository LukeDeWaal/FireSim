import numpy as np
import imageio as im
from typing import Union
from tqdm import tqdm
import threading
import os, datetime


class SimGrid(object):

    def __init__(self, time: int, forest_size: Union[tuple, list, np.array]):

        self.time = int(time)
        self.forest_size = tuple(forest_size)

        # Important Parameters and coefficients
        self.__r_evap = lambda r_grid, k: k/(r_grid+k)  # Base-evaporation rate curve
        self.__weights = np.array([[1/np.sqrt(2), 1, 1/np.sqrt(2)],
                                  [1, 0.5, 1],
                                  [1/np.sqrt(2), 1, 1/np.sqrt(2)]], dtype=float)

        self.wind = {
            'direction': (1,0),
            'magnitude': 0
        }

        self.__retardant_efficiency = 5.0 # Lower numbers will reduce effectiveness
        self.__k = 0.001                  # lower numbers will create less evaporation
        self.__gust_factor = 0            # Higher numbers will make wind more random

        # Main grid where simulation values are stored
        self.main_grid = np.zeros((self.time, *self.forest_size, 3), dtype=float)  # This grid will be used to convert to gif

        # These grids will be temporary grids used for calculations
        self.F_grid = np.random.rand(*self.forest_size)         # Fuel Grid
        self.I_grid = np.zeros(self.forest_size, dtype=float)  # Intensity Grid
        self.I_grid[self.forest_size[0] // 2, self.forest_size[1] // 2] = 1.0
        self.R_grid = np.zeros(self.forest_size, dtype=float)  # Retardant Grid

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

    def clear_space(self, topleft: tuple, size: tuple):
        """
        Make part of the grid unburnable
        :param topleft: top-left coordinate of the grid
        :param size: size of the grid
        """

        self.F_grid[topleft[0]:topleft[0]+size[0], topleft[1]: topleft[1]+size[1]] = np.zeros(size, dtype=float)

    def place_retardant(self, amount: float, topleft: tuple, size: tuple, randomness: float = 0):
        """
        Set a certain part of the grid as retardant
        :param amount: Amount of retardant, 0-1
        :param topleft: top-left coordinate of the grid
        :param size: size of the grid
        """

        retardant = np.ones(size) * amount

        if randomness == 0:
            pass

        else:
            delta_distribution = np.random.uniform(-0.5, 0.5, size)*randomness
            retardant += delta_distribution
            retardant[retardant > 1] = 1
            retardant[retardant < 0] = 0

        self.R_grid[topleft[0]:topleft[0]+size[0], topleft[1]: topleft[1]+size[1]] = retardant

    def set_retardant_efficiency(self, e: Union[float, int]):

        self.__retardant_efficiency = abs(e)

    def set_retardant_evaporation_constant(self, k: float = 0.005):

        self.__k = k

    def set_wind(self, direction: Union[tuple, list, np.array, np.ndarray], magnitude: Union[float, int]):
        """
        Set the general wind direction for the simulation
        :param direction: (steps in x ('downwards'), steps in y ('right'))
        :param magnitude: 0-1
        """
        if len(direction) == 2:
            direction = np.array(direction, dtype=float)
            self.wind['direction'] = direction/np.linalg.norm(direction)
        else:
            raise ValueError("Wrong Shape input for direction")

        if 1 >= magnitude >= 0:
            self.wind['magnitude'] = magnitude/5

        else:
            raise ValueError("Magnitude has to be between 1 and 0")

        self.__weights += magnitude*self.__wind_direction(direction)

    def run(self, name: str = None, path: str = os.getcwd()+'\\simulations\\'):

        if name is None:
            time = datetime.datetime.now()
            date = time.date()
            date = f"{date.month}_{date.day}"
            time = time.time()
            time = f"{time.hour}_{time.minute}_{time.second}"
            name = f"sim_{date}_{time}"

        for t in tqdm(range(self.time)):
            self.__update(t)

        self.__create_coloured_grid()
        self.__save_simulation(name, path)

    @staticmethod
    def __wind_direction(direction: Union[np.array, np.ndarray]):
        """
        Calculate the delta in the weights for the kernel required to model wind direction
        :param direction: 2 by 1 vector
        :return: weights matrix
        """

        weights = np.zeros((3,3))

        for x in range(3):
            for y in range(3):
                weights[x, y] = -direction[0]*(x - 1) - direction[1]*(y-1)

        return weights

    def __initialize_grid(self):
        """
        Implements initial values into main grid
        """

        # Set up main grid
        self.main_grid[0, :, :, 0] = np.array(self.F_grid)
        self.main_grid[0, :, :, 1] = np.array(self.I_grid)
        self.main_grid[0, :, :, 2] = np.array(self.R_grid)

    def __kernel_average(self, grid: Union[np.array, np.ndarray], position: tuple):
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
            size[0] = 2

        elif xrange[1] >= self.forest_size[0]:
            xrange[1] = self.forest_size[0]-1
            weight_idx[0][1] -= 1
            size[0] = 2

        if yrange[0] < 0:
            yrange[0] = 0
            weight_idx[1][0] += 1
            size[1] = 2

        elif yrange[1] >= self.forest_size[0]:
            yrange[1] = self.forest_size[0]-1
            weight_idx[1][1] -= 1
            size[1] = 2

        new_weights = self.__weights[weight_idx[0][0]:weight_idx[0][1], weight_idx[1][0]:weight_idx[1][1]]

        return np.average(grid[xrange[0]:xrange[1]+1, yrange[0]:yrange[1]+1], weights=new_weights)

    def __intensity_averages(self):

        avgs = np.zeros(self.forest_size, dtype=float)

        for x in range(self.forest_size[0]):
            for y in range(self.forest_size[1]):
                avgs[x, y] = self.__kernel_average(self.I_grid, (x, y))

        return avgs

    def __update_fuel(self, t: int):

        old_grid = self.F_grid.copy()

        for x in range(self.forest_size[0]):
            for y in range(self.forest_size[1]):
                if self.R_grid[x, y] >= self.I_grid[x, y]:
                    self.F_grid[x, y] = old_grid[x, y]

                else:
                    self.F_grid[x, y] = old_grid[x, y] + self.R_grid[x, y] - self.I_grid[x, y]

        self.F_grid[self.F_grid < 0.0001] = 0
        self.F_grid[self.F_grid > 1] = 1
        self.main_grid[t, :, :, 0] = self.F_grid
        return old_grid

    def __update_retardant(self, t: int):

        old_grid = self.R_grid.copy()
        # self.R_grid = (1 - (self.__r_evap(self.R_grid)+self.I_grid))*self.R_grid

        evap = self.__r_evap(self.R_grid, k=self.__k)

        for x in range(self.forest_size[0]):
            for y in range(self.forest_size[1]):
                if self.R_grid[x, y] >= self.I_grid[x, y]:
                    self.R_grid[x, y] = (1 - (evap[x, y] + self.I_grid[x, y]/2)) * self.R_grid[x, y]
                else:
                    self.R_grid[x, y] = 0

        self.R_grid[self.R_grid < 0.0001] = 0
        self.R_grid[self.R_grid > 1] = 1
        self.main_grid[t, :, :, 2] = self.R_grid
        return old_grid

    def __update_intensities(self, t: int, old_fuel: Union[np.array, np.ndarray], old_retardant: Union[np.array, np.ndarray]):

        old_grid = self.I_grid.copy()

        delta_fuel = old_fuel - self.F_grid
        delta_retardant = old_retardant - self.R_grid

        iavgs = self.__intensity_averages()

        for x in range(self.forest_size[0]):
            for y in range(self.forest_size[1]):
                if old_fuel[x, y] > 0:
                    self.I_grid[x, y] = old_grid[x, y]*(1+delta_fuel[x,y]) + iavgs[x, y] - delta_retardant[x, y]*self.__retardant_efficiency

                else:
                    self.I_grid[x, y] = old_grid[x, y]/10

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

    def __create_coloured_grid(self):

        for t in tqdm(range(self.time)):
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
                                colour = self.__ground_rgb[int(np.round(cell[0] * len(self.__ground_rgb), 0))]

                            except IndexError:
                                colour = self.__ground_rgb[-1]

                    else:
                        raise ValueError(f"Could not assign colour to grid point {(x, y)} with {[i for i in cell]}")

                    self.coloured[t, x, y, :] = colour

    def __save_simulation(self, name: str, path: str):
        """
        Save simulation as gif
        :param name: name of file
        :param path: path to older in which to store
        """

        im.mimsave((path+f'/{name}.gif'), self.coloured)




if __name__ == '__main__':

    S = SimGrid(100, (300, 300))
    S.place_retardant(0.95, (100, 100), (5, 100), randomness=0.9)
    S.place_retardant(0.95, (100, 100), (100, 5), randomness=0.9)
    S.set_retardant_evaporation_constant(k=0.005)
    S.set_retardant_efficiency(e=1)
    S.set_wind(np.array([-1, -1]), 0.3)
    S.run()
    a = S.main_grid
    b = a[0, :, :, 2]
    c = a[1, :, :, 2]
    d = a[10, :, :, 2]
    i = a[1, :, :, 1]
    i2 = a[10, :, :, 1]

