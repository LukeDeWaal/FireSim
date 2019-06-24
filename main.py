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

    def clear_space(self, topleft: tuple, size: tuple):

        self.simulation.clear_space(topleft, size)

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
    GRID_SIZE = (200, 400)

    S = SimInterface(TIME, GRID_SIZE,
                     maxfuel=4,
                     evaporation=0.00075,
                     efficiency=1.0,
                     wind={
                         'direction': [0, 1],
                         'magnitude': 0.75,
                         'randomness': 0.4
                     },
                     elevation={
                         'direction': [0, 1],
                         'slope_angle': 10 * np.pi / 180,
                         'randomness': 0.5
                     }
                )

    S.set_fire(0.1, (150, 10))

    # S.clear_space((25, 275), (20, 20))
    S.set_dry_patch((0, 50), (299, 500), amount=0.5)

    S.plan_retardant_drops(
        drop_1={
            'time': 150,              # Frame at which to drop
            'velocity': 12,         # Pixels/frame
            'amount': 1.5,            # Average amount per cell; randomisation applies
            'start': [305, 275],      # Starting point of the drop
            'end': [-10, 250],       # Ending point of the drop
            'width': 17              # Width of the retardant line (only uneven numbers work)
        }
        # drop_2={
        #     'time': 175,              # Frame at which to drop
        #     'velocity': 12,         # Pixels/frame
        #     'amount': 1.5,            # Average amount per cell; randomisation applies
        #     'start': [305, 275],      # Starting point of the drop
        #     'end': [-10, 250],       # Ending point of the drop
        #     'width': 17              # Width of the retardant line (only uneven numbers work)
        # }
        # drop_3={
        #     'time': 0,              # Frame at which to drop
        #     'velocity': 10,         # Pixels/frame
        #     'amount': 1.5,            # Average amount per cell; randomisation applies
        #     'start': [210, 60],      # Starting point of the drop
        #     'end': [50, 100],       # Ending point of the drop
        #     'width': 5              # Width of the retardant line (only uneven numbers work)
        # },
        # drop_4 = {
        #     'time': 0,
        #     'velocity': 10,
        #     'amount': 1.5,
        #     'start': [120, 250],
        #     'end': [250, 200],
        #     'width': 5
        # }
    )

    S.run()
    S.show_plots()
