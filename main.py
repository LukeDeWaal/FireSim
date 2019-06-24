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

    def set_fire(self, amount: float, position: tuple):
        self.simulation.set_fire(amount, position)

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

    def set_dry_patch(self, top_left: tuple, bottom_right: tuple, amount: float, randomness=0.3):

        self.simulation.set_dry_patch(top_left, bottom_right, amount, randomness)

    def get_statistics(self):

        self.simulation.plot_progress()


if __name__ == '__main__':

    TIME = 250
    GRID_SIZE = (300, 300)

    S = SimInterface(TIME, GRID_SIZE,
                     maxfuel=4,
                     evaporation=0.00075,
                     efficiency=0.5,
                     wind={
                         'direction': [-1, -1],
                         'magnitude': 0.66,
                         'randomness': 0.5
                     },
                     elevation={
                         'direction': [1, 1],
                         'slope_angle': 0,#30 * np.pi / 180,
                         'randomness': 0.5
                     }
                     )

    S.set_fire(0.1, (GRID_SIZE[0] // 2, GRID_SIZE[1] // 2))
    S.set_fire(0.1, (GRID_SIZE[0] // 2+30, GRID_SIZE[1] // 2+30))

    S.plan_retardant_drops(
        drop_1={
            'time': 50,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.5,            # Average amount per cell; randomisation applies
            'start': [80, 80],      # Starting point of the drop
            'end': [100, 250],       # Ending point of the drop
            'width': 7              # Width of the retardant line (only uneven numbers work)
        },
        drop_2={
            'time': 60,
            'velocity': 10,
            'amount': 1.5,
            'start': [80, 80],
            'end': [250, 100],
            'width': 7
        },
        drop_3={
            'time': 80,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.5,            # Average amount per cell; randomisation applies
            'start': [100, 245],      # Starting point of the drop
            'end': [250, 100],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
        drop_4 = {
            'time': 90,
            'velocity': 10,
            'amount': 1.5,
            'start': [100, 250],
            'end': [80, 80],
            'width': 7
        }
    )

    S.run()
