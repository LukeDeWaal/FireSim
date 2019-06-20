import datetime
import os
from typing import Union
import imageio as im
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.Simulation import SimGrid


class SimulationInterface(object):

    def __init__(self, n: int, time: int, forest_size: Union[tuple, list, np.array]):
        """
        :param n: Number of simulations to run
        :param time: length of simulations
        :param forest_size: Sizes of simulations
        """

        self.simulations = [SimGrid(time, forest_size) for _ in range(n)]  # Container for SimGrid objects
        self.planned_retardants = np.zeros((n, time), dtype=object)

    def time_drops(self,
                   times: Union[np.array, np.ndarray, list],
                   amounts: Union[np.array, np.ndarray, list],
                   lengths: Union[np.array, np.ndarray, list],
                   p_0: Union[np.array, np.ndarray, list],
                   v: Union[np.array, np.ndarray, list],
                   randomnesses: Union[np.array, np.ndarray, list],
                   kernel_shapes: Union[np.array, np.ndarray, list]):
        """
        Place Retardant along a slanted line with a certain width
        :param times:
        :param amounts:
        :param lengths:
        :param p_0:
        :param v:
        :param randomnesses:
        :param kernel_shapes:
        :return:
        """

        for sim, t, amount, length, p_i, v_i, shape, randomness in zip(self.simulations, times, amounts, lengths, p_0, v, kernel_shapes, randomnesses):
            sim.retardant_along_line(t, amount, length, p_i, v_i, shape=shape, randomness=randomness)

    def place_retardants(self, amounts: list, toplefts: list, sizes: list, randomnesses: list, linetype: str = 'line or grid'):
        """
        Method to place retardants in simulations
        :param amounts: [ amount_1, amount_2, ... , amount_n]
        :param toplefts: [ (topx, topy)_1, (topx, topy)_2, ... , (topx, topy)_n]
        :param sizes: [ (lenx, leny)_1, (lenx, leny)_2, ... , (lenx, leny)_n]
        :param randomnesses: [ r_1, r_2, ..., r_n ]
        """

        for sim, amount, topleft, size, rand in zip(self.simulations, amounts, toplefts, sizes, randomnesses):
            sim.place_retardant(amount, topleft, size, randomness=rand)

    def set_retardant_efficiencies(self, e: list):
        """
        Set retardant efficiencies for the simulations
        :param e: [ e_1, e_2, ..., e_n ]
        """

        for sim, ei in zip(self.simulations, e):
            sim.set_retardant_efficiency(e=ei)

    def set_winds(self, winds: list, randomnesses: list):
        """
        Set the winds for the simulations
        :param winds: [ ([vx, vy], mag)_1, ([vx, vy], mag)_2, ..., ([vx, vy], mag)_n ]
        :param randomnesses: [ r_1, r_2, ..., r_n ]
        """

        for sim, wind, randomness in zip(self.simulations, winds, randomnesses):
            sim.set_wind(direction=wind[0], magnitude=wind[1], randomness=randomness)

    def set_elevations(self, elevations: list, randomnesses: list):
        """
        Set the elevations for the simulation
        :param elevations: [ ([dx, dy], theta)_1, ([dx, dy], theta)_2, ..., ([dx, dy], theta)_n
        :param randomnesses: [ r_1, r_2, ..., r_n ]
        """

        for sim, elevation, rand in zip(self.simulations, elevations, randomnesses):
            sim.set_elevation(elevation[0], elevation[1], randomness=rand)

    def set_evaporation_constants(self, k: list):
        """
        Set retardant evaporation constants for the simulations
        :param e: [ k_1, k_2, ..., k_n ]
        """

        for sim, ki in zip(self.simulations, k):
            sim.set_retardant_evaporation_constant(k=ki)

    def clear_spaces(self, spaces: list):
        """
        Clear a space on the grid from fuel
        :param spaces: [ ( [(topx, topy), (sizex, sizey)] )_1, [(topx, topy), (sizex, sizey)] )_2, ..., [(topx, topy), (sizex, sizey)] )_n ]
        """

        for sim, space in zip(self.simulations, spaces):
            sim.clear_space(space[0], space[1])

    def run_simulations(self):
        """
        Run the simulation
        """

        for sim in self.simulations:
            sim.run()

    def plot_statistics(self):

        for sim in self.simulations:
            sim.plot_progress()
            plt.show()


if __name__ == '__main__':

    N_SIMULATIONS = 2
    TIME = 150
    GRID_SIZE = (200, 200)

    RETARDANTS = {
        'amounts': [
            0.99,
            0.7
        ],
        'toplefts': [
            (65, 0),
            (0, 120)
        ],
        'sizes': [
            (25, 200),
            (100, 25)
        ],
        'randomnesses': [
            0.66,
            0.66
        ]
    }
    ELEVATIONS = {
        'elevations': [
            ([-1, 0], np.pi / 6),
            ([-1, -1], np.pi / 4)
        ],
        'randomnesses': [
            0.0,
            0.0
        ]
    }
    EVAPORATION_CONSTANTS = {
        'k': [
            0.001,
            0.002
        ]
    }
    RETARDANT_EFFICIENCIES = {
        'e': [
            0.2,
            0.15
        ]
    }
    WINDS = {
        'winds': [
            (np.array([-1, -1]), 0.75),
            (np.array([-1, -1]), 0.0)
        ],
        'randomnesses': [
            1.0,
            0.5
        ]
    }

    S = SimulationInterface(N_SIMULATIONS, TIME, GRID_SIZE)
    S.set_elevations(**ELEVATIONS)
    S.set_evaporation_constants(**EVAPORATION_CONSTANTS)
    S.set_retardant_efficiencies(**RETARDANT_EFFICIENCIES)
    S.time_drops([25, 2], [100, 100], [100]*2, [np.array([80, 200], dtype=np.int)]*2, [np.array([0, -1])]*2, [0.4]*2, kernel_shapes=[(9, 9), (5, 5)])
    S.set_winds(**WINDS)
    S.place_retardants(**RETARDANTS)
    S.run_simulations()
    S.plot_statistics()