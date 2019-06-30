from main import SimInterface
import numpy as np

TIME = 300
GRID_SIZE = (300, 300)

Sim = SimInterface(TIME, GRID_SIZE,
                     maxfuel=4,
                     evaporation=0.00095,
                     efficiency=1.5,
                     wind={
                         'direction': [1, -1],
                         'magnitude': 0.75,
                         'randomness': 0.4
                     },
                     elevation={
                         'direction': [-1, 1],
                         'slope_angle': 10 * np.pi / 180,
                         'randomness': 0.5
                     }
                )

Sim.clear_space((10, 250), (20, 40))

for i in range(5):
    Sim.set_fire(0.1, (GRID_SIZE[0]//2 + np.random.randint(-5, 5), GRID_SIZE[1]//2 + np.random.randint(-5, 5)))

Sim.plan_retardant_drops(
        drop_1={
            'time': 0,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.25,            # Average amount per cell; randomisation applies
            'start': [30, 150],      # Starting point of the drop
            'end': [150, 30],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
    drop_2={
            'time': 0,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.25,            # Average amount per cell; randomisation applies
            'start': [30, 150],      # Starting point of the drop
            'end': [150, 30],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
    }
)

Sim.run()