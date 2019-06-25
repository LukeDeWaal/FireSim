from main import SimInterface
import numpy as np

TIME = 400
GRID_SIZE = (200, 400)

Sim = SimInterface(TIME, GRID_SIZE,
                     maxfuel=4,
                     evaporation=0.00075,
                     efficiency=0.75,
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


for i in range(GRID_SIZE[0]):
    Sim.set_fire(0.1, (i, np.random.randint(0, 5)))

Sim.plan_retardant_drops(
        drop_1={
            'time': 120,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.95,            # Average amount per cell; randomisation applies
            'start': [GRID_SIZE[0], 120],      # Starting point of the drop
            'end': [100-10, 120],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
        drop_2={
            'time': 125,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.95,            # Average amount per cell; randomisation applies
            'start': [2*GRID_SIZE[0]//3, 123],      # Starting point of the drop
            'end': [-10, 123],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
        drop_3={
            'time': 153,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.75,            # Average amount per cell; randomisation applies
            'start': [GRID_SIZE[0], 128],      # Starting point of the drop
            'end': [100-10, 128],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        },
        drop_4={
            'time': 158,              # Frame at which to drop
            'velocity': 10,         # Pixels/frame
            'amount': 1.75,            # Average amount per cell; randomisation applies
            'start': [2*GRID_SIZE[0]//3, 131],      # Starting point of the drop
            'end': [-10, 131],       # Ending point of the drop
            'width': 5              # Width of the retardant line (only uneven numbers work)
        }
)

Sim.run()