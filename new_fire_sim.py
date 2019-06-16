import numpy as np
import imageio as im
from typing import Union
from tqdm import tqdm
import threading as th
from multiprocess import Pool
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

        self.elevation = {
            'direction': (1, 0),
            'slope': 0.0
        }

        self.__retardant_efficiency = 5.0 # Lower numbers will reduce effectiveness
        self.__k = 0.001                  # lower numbers will create less evaporation
        self.__gust_factor = 0            # Higher numbers will make wind more random

        # Main grid where simulation values are stored
        self.main_grid = np.zeros((self.time, *self.forest_size, 4), dtype=float)  # This grid will be used to convert to gif

        # These grids will be temporary grids used for calculations
        self.F_grid = np.random.rand(*self.forest_size)         # Fuel Grid
        self.I_grid = np.zeros(self.forest_size, dtype=float)  # Intensity Grid
        self.I_grid[self.forest_size[0] // 2, self.forest_size[1] // 2] = 1.0
        self.R_grid = np.zeros(self.forest_size, dtype=float)  # Retardant Grid
        self.H_grid = np.zeros(self.forest_size, dtype=float)  # Elevation grid

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

        self.F_grid[topleft[0]:topleft[0]+size[0], topleft[1]: topleft[1]+size[1]] = np.zeros(size, dtype=float)

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
            delta_distribution = np.random.uniform(-amount, amount, size)*randomness
            retardant += delta_distribution
            retardant[retardant > 1] = 1
            retardant[retardant < 0] = 0

        self.R_grid[topleft[0]:topleft[0]+size[0], topleft[1]: topleft[1]+size[1]] = retardant

    def set_retardant_efficiency(self, e: Union[float, int]):

        self.__retardant_efficiency = abs(e)

    def set_retardant_evaporation_constant(self, k: float = 0.005):

        self.__k = k

    def set_wind(self, direction: Union[tuple, list, np.array, np.ndarray], magnitude: Union[float, int], randomness: float = 0.2):
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

        self.__weights += magnitude*self.__directed_weights_increase(direction)

        self.__gust_factor = randomness

    def set_elevation(self, slope_direction: Union[tuple, list, np.array, np.ndarray], slope_angle: float):

        if len(slope_direction) == 2:
            direction = np.array(slope_direction, dtype=float)
            self.elevation['direction'] = direction/np.linalg.norm(direction)
        else:
            raise ValueError("Wrong Shape input for direction")

        if np.pi/2 > slope_angle >= 0:
            self.elevation['slope'] = slope_angle

        else:
            raise ValueError("Magnitude has to be between 1 and 0")

        self.__weights += magnitude*self.__directed_weights_increase(direction)

    def run(self, name: str = None, path: str = None):

        if path is None:

            path = os.getcwd() + '\\simulations\\'

        if name is None:
            time = datetime.datetime.now()
            date = time.date()
            date = f"{date.month}_{date.day}"
            time = time.time()
            time = f"{time.hour}_{time.minute}_{time.second}"
            name = f"sim_{date}_{time}"

        print("===== SIMULATING =====")
        for t in tqdm(range(self.time)):
            self.__update(t)

        print("===== COLOURING =====")
        self.__create_coloured_grid()
        self.__save_simulation(name, path)

    @staticmethod
    def __directed_weights_increase(direction: Union[np.array, np.ndarray]):
        """
        Calculate the delta in the weights for the kernel
        :param direction: 2 by 1 vector
        :return: delta weights matrix
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
        self.main_grid[0, :, :, 3] = np.array(self.H_grid)

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

        new_weights = self.__weights[weight_idx[0][0]:weight_idx[0][1], weight_idx[1][0]:weight_idx[1][1]] + \
                      self.__random_kernel(tuple(size), (-self.__gust_factor, self.__gust_factor))

        return np.average(grid[xrange[0]:xrange[1]+1, yrange[0]:yrange[1]+1], weights=new_weights)

    @staticmethod
    def __random_kernel(size: tuple, randrange: tuple = (-0.2, 0.2)):

        return np.random.uniform(*randrange, size)

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


class SimulationInterface(object):

    def __init__(self, n: int, time: int, forest_size: Union[tuple, list, np.array]):

        self.simulations = [SimGrid(time, forest_size) for _ in range(n)]

    def place_retardants(self, amounts: list, toplefts: list, sizes: list, randomnesses: list):

        for sim, amount, topleft, size, rand in zip(self.simulations, amounts, toplefts, sizes, randomnesses):
            sim.place_retardant(amount, topleft, size, randomness=rand)

    def set_retardant_efficiencies(self, e: list):

        for sim, ei in zip(self.simulations, e):
            sim.set_retardant_efficiency(e=ei)

    def set_winds(self, winds: list, randomnesses: list):

        for sim, wind, randomness in zip(self.simulations, winds, randomnesses):
            sim.set_wind(direction=wind[0], magnitude=wind[1], randomness=randomness)

    def set_evaporation_constants(self, k: list):

        for sim, ki in zip(self.simulations, k):
            sim.set_retardant_evaporation_constant(k=ki)

    def clear_spaces(self, spaces: list):

        for sim, space in zip(self.simulations, spaces):
            sim.clear_space(space[0], space[1])

    def run_simulations(self):

        for sim in self.simulations:
            sim.run()




if __name__ == '__main__':

    N_SIMULATIONS = 2
    TIME = 100
    GRID_SIZE = (300, 300)

    RETARDANTS = {
        'amounts': [
            0.5,
            0.7
        ],
        'toplefts': [
            (100, 100),
            (140, 140)
        ],
        'sizes': [
            (15, 100),
            (100, 5)
        ],
        'randomnesses': [
            0.1,
            0.5
        ]
    }

    S = SimulationInterface(N_SIMULATIONS, TIME, GRID_SIZE)
    S.set_evaporation_constants(k=[
        0.01,
        0.005
    ])
    S.set_retardant_efficiencies(e=[
        3.5,
        2.5
    ])
    S.set_winds(
        winds=[
        (np.array([-1, -1]), 0.5),
        (np.array([-1, 1]), 0.5)
    ],
        randomnesses=[
        0.1,
        0.5
    ])
    S.place_retardants(**RETARDANTS)
    S.run_simulations()