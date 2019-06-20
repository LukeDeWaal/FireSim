from typing import Union

import numpy as np

from src.Simulation import SimGrid


class SimInterface(object):

    def __init__(self, t: int, shape: tuple, maxfuel: float = 5.0, **kwargs):

        self.time = t
        self.shape = shape
        self.simulation = SimGrid(time=t, forest_size=shape, maxfuel=maxfuel)

        for key, value in kwargs.items():

            if key == 'wind':
                self.set_simulation_wind(**value)

            elif key == 'elevation':
                self.set_grid_elevation(**value)

            elif key == 'evaporation':
                self.set_retardant_evaporation_constant(value)

            elif key == 'efficiency':
                self.set_retardant_efficiency(value)

    def run(self, name: str = None, path: str = None):

        self.simulation.run(name, path)

    def plan_retardant_drops(self, **drops):

        for ID, value in drops.items():

            if 'width' in value.keys():

                if type(value['width']) in [list, tuple, np.array, np.ndarray]:
                    shape = value['width']

                elif type(value['width']) == int:
                    shape = (value['width'], value['width'])

                else:
                    raise TypeError("Expected an integer or 2-item iterable")

            else:
                shape = (3, 3)

            if 'randomness' not in value.keys():
                value['randomness'] = 0.5

            self.simulation.retardant_along_line(t=value['time'],
                                                 amount=value['amount'],
                                                 length=value['length'],
                                                 p_0=value['start'],
                                                 v=value['direction'],
                                                 shape=shape,
                                                 randomness=value['randomness'])

    def set_retardant_efficiency(self,
                                 eta: Union[float, int]):

        self.simulation.set_retardant_efficiency(e=eta)

    def set_retardant_evaporation_constant(self,
                                           k: float):

        self.simulation.set_retardant_evaporation_constant(k=k)

    def set_simulation_wind(self,
                            direction: Union[np.array_equal, np.ndarray, tuple, list],
                            magnitude: float,
                            randomness: float = 0.5):

        self.simulation.set_wind(direction=direction, magnitude=magnitude, randomness=randomness)

    def set_grid_elevation(self,
                           direction: Union[np.array_equal, np.ndarray, tuple, list],
                           slope_angle: float,
                           randomness: float = 0.5):

        self.simulation.set_elevation(slope_direction=direction, slope_angle=slope_angle, randomness=randomness)


if __name__ == '__main__':

    TIME = 350
    GRID_SIZE = (300, 300)

    # RETARDANTS = {
    #     'amounts': [
    #         0.99,
    #         0.7
    #     ],
    #     'toplefts': [
    #         (65, 0),
    #         (0, 120)
    #     ],
    #     'sizes': [
    #         (25, 200),
    #         (100, 25)
    #     ],
    #     'randomnesses': [
    #         0.66,
    #         0.66
    #     ]
    # }
    # ELEVATIONS = {
    #     'elevations': [
    #         ([-1, 0], np.pi / 6),
    #         ([-1, -1], np.pi / 4)
    #     ],
    #     'randomnesses': [
    #         0.0,
    #         0.0
    #     ]
    # }
    # EVAPORATION_CONSTANTS = {
    #     'k': [
    #         0.001,
    #         0.002
    #     ]
    # }
    # RETARDANT_EFFICIENCIES = {
    #     'e': [
    #         0.2,
    #         0.15
    #     ]
    # }
    # WINDS = {
    #     'winds': [
    #         (np.array([-1, -1]), 0.75),
    #         (np.array([-1, -1]), 0.0)
    #     ],
    #     'randomnesses': [
    #         1.0,
    #         0.5
    #     ]
    # }
    #
    # S = SimulationInterface(N_SIMULATIONS, TIME, GRID_SIZE)
    # S.set_elevations(**ELEVATIONS)
    # S.set_evaporation_constants(**EVAPORATION_CONSTANTS)
    # S.set_retardant_efficiencies(**RETARDANT_EFFICIENCIES)
    # S.time_drops([25, 2], [100, 100], [100]*2, [np.array([80, 200], dtype=np.int)]*2, [np.array([0, -1])]*2, [0.4]*2, kernel_shapes=[(9, 9), (5, 5)])
    # S.set_winds(**WINDS)
    # S.place_retardants(**RETARDANTS)
    # S.run_simulations()
    # S.plot_statistics()

    S = SimInterface(TIME, GRID_SIZE,
                     maxfuel=4,
                     evaporation=0.00075,
                     efficiency=0.25,
                     wind={
                         'direction': [-1, -1],
                         'magnitude': 0.33,
                         'randomness': 0.5
                     },
                     elevation={
                         'direction': [1, 1],
                         'slope_angle': 30 * np.pi / 180,
                         'randomness': 0.5
                     }
                     )

    S.plan_retardant_drops(
        drop_1={
            'time': 10,
            'direction': [0, 1],
            'amount': 75,
            'start': [90, 50],
            'length': 200,
            'width': 5
        },
        drop_2={
            'time': 20,
            'direction': [0, 1],
            'amount': 75,
            'start': [85, 25],
            'length': 250,
            'width': 7
        },
        drop_3={
            'time': 40,
            'direction': [0, 1],
            'amount': 150,
            'start': [75, 0],
            'length': 200,
            'width': 10
        },
        drop_4={
            'time': 100,
            'direction': [0, 1],
            'amount': 100,
            'start': [40, 0],
            'length': 200,
            'width': 5
        }
    )

    S.run()
