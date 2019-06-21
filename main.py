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

    def show_plots(self):
        self.simulation.plot_progress()

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
                                                 velocity=value['velocity'],
                                                 p_0=value['start'],
                                                 p_1=value['end'],
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

    TIME = 150
    GRID_SIZE = (300, 300)

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
            'time': 0,
            'velocity': 10,
            'amount': 1,
            'start': [90, 50],
            'end': [180, 250],
            'width': 5
        }
        # drop_2={
        #     'time': 20,
        #     'velocity': 10,
        #     'amount': 30,
        #     'start': [85, 50],
        #     'end': [85, 250],
        #     'width': 5
        # },
        # drop_3={
        #     'time': 60,
        #     'velocity': 5,
        #     'amount': 50,
        #     'start': [70, 50],
        #     'end': [70, 250],
        #     'width': 5
        # }
    )

    S.run()
